import typing
from copy import deepcopy
from typing import Sequence, Generator, Any

import torch

from fluke import FlukeENV
from fluke.evaluation import Evaluator
from fluke.comm import Message
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.server import EarlyStopping
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict
from playground.centralizedSL import CentralizedSL
from playground.clientSL import ClientSL
from playground.serverSL import ServerSL

class P2PSL(CentralizedSL):
    def get_client_class(self):
        return ClientP2PSL

    def get_server_class(self):
        return ServerP2PSL

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        with FlukeENV().get_live_renderer():
            progress_sl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            task_rounds = progress_sl.add_task("[red]SL Rounds", total=n_rounds * client_x_round)
            task_local = progress_client.add_task("[green]Client Updates", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            self._round_zero()
            for rnd in range(self.rounds, total_rounds):
                try:
                    self.notify(event="start_round", round=rnd + 1,
                                global_model=torch.nn.Sequential(self.server.client_model, self.server.model))

                    eligible = typing.cast(Sequence[ClientP2PSL], self.server.get_eligible_clients(eligible_perc)) #eligible is still of type Sequence[Client] but the type checker knows that the return type is Sequence[ClientP2PSL]


                    self.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                    for c, client in enumerate(eligible):
                        first_client_first_round = c == 0 and rnd == 0
                        self.server.send_last_client_trained_info(client.index, first_client_first_round=first_client_first_round)

                        if first_client_first_round:
                            client.model = deepcopy(self.server.client_model)
                        else:
                            eligible[c - 1].send_model(client.index)

                        client.start_round(rnd + 1)
                        for _ in range(self.hyper_params.client.local_epochs):
                            local_update = client.train_epoch()
                            for _ in local_update:
                                self.server.train_on_smashed_data(client.index)
                            self.server.end_epoch()

                        client.end_round(rnd + 1)
                        self.server.end_round(client.index)
                        progress_client.update(task_id=task_local, completed=c + 1)
                        progress_sl.update(task_id=task_rounds, advance=1)

                    self._compute_evaluation_full_model(rnd)
                    self.notify(event="end_round", round=rnd + 1)
                    self.rounds += 1

                    path, freq, g_only = FlukeENV().get_save_options()
                    if freq > 0 and rnd % freq == 0:
                        self.save(path, g_only, rnd)

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

class ClientP2PSL(ClientSL):
    def send_model(self, mbox: str | int = "server") -> None:
        self.channel.send(Message(self.model, "client_model", self.index, inmemory=True), mbox)

    def receive_model(self, sender: str | int ="server") -> None:
        msg = self.channel.receive(self.index, sender, msg_type="client_model")
        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def receive_client_info(self) -> int:
        msg = self.channel.receive(self.index, "server", msg_type="last_client_index")
        return msg.payload

    def start_round(self, current_round: int):
        self.n_batches = 0
        self.running_loss = 0.0
        self.local_smashed = None
        self._load_from_cache()
        sender, server_lr = self.receive_client_info()  # unpack the (sender, lr) tuple
        self._server_lr = server_lr
        if sender is not None:
            self.receive_model(sender)
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, _ = self._optimizer_cfg(self.model)

        for pg in self.optimizer.param_groups:  # adopt server LR, EVERY round
            pg["lr"] = self._server_lr

        self.notify("start_fit", round=current_round, client_id=self.index, model=self.model)

    def end_round(self, current_round) -> None:
        self._last_round = current_round

        self.notify(
            "end_fit",
            round=current_round,
            client_id=self.index,
            model=self.model,
            loss=(self.running_loss / max(1, self.n_batches)),
        )

        self.model.cpu()
        clear_cuda_cache()
        # self.send_model()
        self._check_persistency()
        self._save_to_cache()

class ServerP2PSL(ServerSL):
    def __init__(
            self,
            model: torch.nn.Module,  # modello server-side
            client_model: torch.nn.Module,  # modello client-side  (solo nel caso centralized)
            test_set: FastDataLoader | None,
            clients: Sequence[ClientSL],
            optimizer_cfg: OptimizerConfigurator,
            loss_fn: torch.nn.Module,
            weighted: bool = False,
            lr: float = 1.0,
            clipping: float = 0.0,
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

        self.last_client_trained_index = None

    def send_last_client_trained_info(self, client_index: int, first_client_first_round=False) -> None:
        if self.optimizer is None:  # same lazy-init guard as CentralizedSL
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
        current_lr = self.optimizer.param_groups[0]["lr"]
        last_idx = None if first_client_first_round else self.last_client_trained_index
        self.channel.send(
            Message((last_idx, current_lr), "last_client_index", "server", inmemory=True),
            client_index,
        )

    def end_round(self, client_index):
        self.last_client_trained_index = client_index
        self.model.cpu()
        clear_cuda_cache()

    def evaluate_full_model(self, evaluator: Evaluator, round: int) -> dict[str, float]:
        # "concateno" le due reti per valutare il modello completo
        if self.test_set is not None:
            full_model = torch.nn.Sequential(
                self.clients[-1].model,
                self.model
            )
            return evaluator.evaluate(round, full_model, self.test_set, loss_fn=None, device=self.device)
        return {}