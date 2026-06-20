import math
import typing

import torch

from fluke import FlukeENV, DDict
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader, DataSplitter
from fluke.server import EarlyStopping
from fluke.utils import clear_cuda_cache
from playground.psl import ClientPSL, ServerPSL
from playground.splitfedv2 import SplitFedV2

from datetime import datetime

class ServerEPSL(ServerPSL):
    def __init__(
            self,
            model: torch.nn.Module,  # modello server-side
            client_model: torch.nn.Module,  # modello client-side  (solo nel caso centralized)
            test_set: FastDataLoader | None,
            clients: typing.Sequence[ClientPSL],
            optimizer_cfg: OptimizerConfigurator,
            loss_fn: torch.nn.Module,
            weighted: bool = False,
            lr: float = 1.0,
            clipping: float = 0.0,
            phi: float = 0.5,
            **kwargs,
    ):
        super().__init__(
            model=model,
            client_model=client_model,
            test_set=test_set,
            clients=clients,
            optimizer_cfg=optimizer_cfg,
            weighted=weighted,
            loss_fn=loss_fn,
            lr=lr,
            clipping=clipping,
            **kwargs,
        )
        self.phi = phi

    def end_client_round(self, client_index=None):
        self.model.cpu()
        clear_cuda_cache()

    def server_step(self, client_indices: list[int]) -> None:
        self.model.train()
        self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        smashed_data = {ci: self.receive_smashed_data(ci) for ci in client_indices}
        batch_sizes = [smashed_data[ci][0].shape[0] for ci in client_indices]
        C = len(client_indices)

        smashed_per_client = [
            smashed_data[ci][0].to(self.device).detach().requires_grad_(True)
            for ci in client_indices
        ]
        smashed_list = torch.cat(smashed_per_client, dim=0)
        y_list = torch.cat([smashed_data[ci][1].to(self.device) for ci in client_indices], dim=0)

        self.optimizer.zero_grad()
        server_output = self.model(smashed_list)
        loss = self.hyper_params.loss_fn(server_output, y_list)

        last_layer_grads = torch.autograd.grad(loss, server_output, retain_graph=True)[0]
        g_per_client = list(torch.split(last_layer_grads, batch_sizes, dim=0))

        k = math.ceil(self.phi * min(batch_sizes))

        by_index = {c.index: c for c in self.clients}
        lambdas = torch.tensor(
            self._get_client_weights([by_index[ci] for ci in client_indices]),
            device=self.device, dtype=last_layer_grads.dtype,
        )

        if k > 0:
            head_stack = torch.stack([g_per_client[i][:k] for i in range(C)], dim=0)  # (C,k,·)
            agg_head = (head_stack * lambdas.view(C, 1, 1)).sum(dim=0)  # (k,·)

        effective = []
        for i in range(C):
            tail = g_per_client[i][k:]
            if k == 0:
                effective.append(tail)
            elif i == 0:
                effective.append(torch.cat([agg_head, tail], dim=0))
            else:
                effective.append(torch.cat([torch.zeros_like(g_per_client[i][:k]), tail], dim=0))
        effective_grad = torch.cat(effective, dim=0)

        server_output.backward(effective_grad)
        self._clip_grads()
        self.optimizer.step()

        shared_cut_head = smashed_per_client[0].grad[:k] if k > 0 else None
        loss_val = float(loss.item())
        for i, ci in enumerate(client_indices):
            own = smashed_per_client[i].grad
            full_cut = own if k == 0 else torch.cat([shared_cut_head, own[k:]], dim=0)
            self.send_gradients(full_cut.detach().cpu(), loss_val, ci)

class ClientEPSL(ClientPSL):
    def train_epoch(self) -> typing.Generator:
        for X, y in self.train_set:
            X = X.to(self.device)
            self.optimizer.zero_grad()
            self.local_smashed = self.model(X)
            remote_smashed = self.local_smashed.clone().detach().requires_grad_(True)
            self.send_smashed_data(remote_smashed, y)

            try:
                yield
            except GeneratorExit:
                # Generator closed mid-batch (zip stopped because another client
                # ran out of batches). Server hasn't processed our smashed data
                # this iteration, so there are no gradients to receive. Bail out.
                return

            # Normal resumption: server has sent gradients for the batch we just yielded on.
            grad_cut, server_loss, server_lr = self.receive_gradients()
            for pg in self.optimizer.param_groups:
                pg["lr"] = server_lr
            self.local_smashed.backward(grad_cut.to(self.local_smashed.device))
            self._clip_grads(self.model)
            self.optimizer.step()
            self.running_loss += server_loss
            self.n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()


class EPSL(SplitFedV2):
    def __init__(
            self,
            n_clients: int,
            data_splitter: DataSplitter,
            hyper_params: DDict | dict[str, typing.Any],
            clients: list[ClientPSL] = None,
            server: ServerEPSL = None,
            **kwargs,
    ):
        super().__init__(
            n_clients=n_clients,
            data_splitter=data_splitter,
            hyper_params=hyper_params,
            clients=clients,
            server=server,
            **kwargs,
        )


    def get_client_class(self):
        return ClientEPSL

    def get_server_class(self):
        return ServerEPSL

    def _round_zero(self, eligible_perc) -> None:
        eligible = typing.cast(typing.Sequence[ClientPSL], self.server.get_eligible_clients(eligible_perc))
        self.server.broadcast_model(eligible)

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        self.server = typing.cast(ServerEPSL, self.server)
        with FlukeENV().get_live_renderer():
            progress_sl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            task_rounds = progress_sl.add_task("[red]SL Rounds", total=n_rounds * client_x_round)
            task_local = progress_client.add_task("[green]Client Updates", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            self._round_zero(eligible_perc)
            for rnd in range(self.rounds, total_rounds):
                try:
                    self.notify(event="start_round", round=rnd + 1,
                                global_model=torch.nn.Sequential(self.server.client_model, self.server.model))

                    eligible = typing.cast(typing.Sequence[ClientPSL], self.server.get_eligible_clients(eligible_perc)) # eligible is still of type Sequence[Client] but the type checker knows that the return type is Sequence[ClientPSL]

                    self.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                    active = {}
                    for client in eligible:
                        client.start_round(rnd + 1)
                        gen = client.train_epoch()
                        try:
                            next(gen)  # prime: sends batch 1, yields
                            active[client.index] = gen
                        except StopIteration:
                            pass  # client had no batches at all

                    while active:
                        # Server only sees clients that have smashed data in flight right now
                        self.server.server_step(list(active.keys()))

                        # Advance each live generator: receives prev grad, sends next batch (or finishes)
                        exhausted = []
                        for ci, gen in active.items():
                            try:
                                next(gen)
                            except StopIteration:
                                # Generator exited cleanly — its last grad was received inside this next() call
                                exhausted.append(ci)
                        for ci in exhausted:
                            del active[ci]

                    for client in eligible:
                        client.end_round(rnd + 1)
                    self.server.end_client_round()

                    for c, _ in enumerate(eligible):
                        progress_client.update(task_id=task_local, completed=c + 1)
                    progress_sl.update(task_id=task_rounds, advance=1)

                    # we choose a random client for evaluation
                    random_client = eligible[0]
                    random_client.send_model()
                    self.server.receive_client_model(random_client.index)
                    self.server.end_round()
                    self._compute_evaluation_full_model(rnd)
                    self.notify(event="end_round", round=rnd + 1)
                    self.rounds += 1
                except KeyboardInterrupt:
                    self.notify(event="interrupted")
                    break

                except EarlyStopping:
                    self.notify(event="early_stop", round=self.rounds + 1)
                    break

            progress_sl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        if finalize:
            self.finalize()

        self.notify(event="finished", round=self.rounds + 1)