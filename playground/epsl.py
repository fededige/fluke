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

        smashed_data = {client_index : self.receive_smashed_data(client_index) for client_index in client_indices}

        # 3) Server-side Model Forward Propagation
        #Concateno per formare S(t) e y^(t)
        smashed_list = torch.cat([smashed_data[i][0].to(self.device).requires_grad_(True) for i in client_indices])
        y_list = torch.cat([smashed_data[i][1].to(self.device) for i in client_indices])

        self.optimizer.zero_grad()

        #Forward propagation
        server_output = self.model(smashed_list)

        # 4) Gradient Aggregation and Server-side Model Back Propagation:
        loss = self.hyper_params.loss_fn(server_output, y_list)


        last_layer_activation_grads = torch.autograd.grad(
            outputs=loss,
            inputs=server_output,
            retain_graph=True,
            create_graph=False,
        )[0]


        b = smashed_data[client_indices[0]][0].shape[0]  # per-client batch size
        C = len(client_indices)
        agg_count = math.ceil(self.phi * b)  # ⌈φb⌉

        # Reshape to (C, b, out_dim) — one chunk per client
        grad_per_client = last_layer_activation_grads.view(C, b, -1)

        agg_grads = grad_per_client[:, :agg_count, :]  # (C, ⌈φb⌉, out_dim)
        unagg_grads = grad_per_client[:, agg_count:, :]  # (C, b-⌈φb⌉, out_dim)

        lambdas = torch.tensor(self._get_client_weights([c for c in self.clients if c.index in client_indices])).to(self.device)

        aggregated = (agg_grads * lambdas.view(C, 1, 1)).sum(dim=0)

        aggregated_expanded = aggregated.unsqueeze(0).expand(C, -1, -1)

        modified_grads_per_client = torch.cat([aggregated_expanded, unagg_grads], dim=1)  # (C, b, out_dim)
        modified_last_layer_grads = modified_grads_per_client.reshape(C * b, -1)  # (bC, out_dim)

        cut_layer_grads = torch.autograd.grad(
            outputs=server_output,
            inputs=smashed_list,
            grad_outputs=modified_last_layer_grads,
            retain_graph=True,
        )[0]

        server_output.backward(gradient=modified_last_layer_grads)
        self.optimizer.step()

        # Split cut_layer_grads back per client
        cut_layer_grads_per_client = cut_layer_grads.view(C, b, -1)  # (C, b, q)

        # Aggregated cut grads are the same for all clients, just take from index 0
        aggregated_cut_grads = cut_layer_grads_per_client[0, :agg_count, :]  # (agg_count, q)

        # Store the original spatial shape at the top of server_step
        original_smashed_shape = smashed_data[client_indices[0]][0].shape  # (b, 64, 8, 8)

        # Then when sending, reshape back before sending
        for idx, client_index in enumerate(client_indices):
            unagg_cut_grads = cut_layer_grads_per_client[idx, agg_count:, :]  # (b-agg_count, 4096)
            full_cut_grads = torch.cat([aggregated_cut_grads, unagg_cut_grads], dim=0)  # (b, 4096)

            # Reshape back to original 4D shape before sending
            full_cut_grads = full_cut_grads.reshape(
                b, *original_smashed_shape[1:]
            )  # (b, 64, 8, 8)

            self.send_gradients(full_cut_grads.detach().cpu(), float(loss.item()), client_index)


class ClientEPSL(ClientPSL):
    def train_epoch(self) -> typing.Generator:
        for X, y in self.train_set:
            X = X.to(self.device)
            self.optimizer.zero_grad()
            self.local_smashed = self.model(X)
            remote_smashed = self.local_smashed.clone().detach().requires_grad_(True)
            self.send_smashed_data(remote_smashed, y)

            try:
                yield  # aggiungere commento
            finally:
                grad_cut, server_loss = self.receive_gradients()
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

                    local_updates = {} #list of generators
                    for client in eligible:
                        client.start_round(rnd + 1)
                        local_updates[client.index] = client.train_epoch()

                    # local_updates needs to be unpacked (using *) because it's a single list
                    for _ in zip(*local_updates.values()):
                        self.server.server_step([c.index for c in eligible])

                    for gen in local_updates.values():
                        gen.close()

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