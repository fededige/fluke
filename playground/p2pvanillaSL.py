from typing import Sequence, Generator

import torch

from fluke import FlukeENV
from fluke.comm import Message
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.server import EarlyStopping
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict
from playground.centralizedSL import CentralizedSL
from playground.clientSL import ClientSL
from playground.serverSL import ServerSL

class P2PCentralizedSL(CentralizedSL):
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

                    eligible = self.server.get_eligible_clients(eligible_perc)

                    self.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                    for c, client in enumerate(eligible):
                        if c == 0 and rnd == 0:
                            self.server.send_client_model(client.index)
                        else:
                            self.server.send_last_client_trained_info(client.index)
                            eligible[c - 1].send_model(client.index) #if c == 0 and rnd != 0 => eligible[0 - 1] == eligible[len(eligible) - 1]

                        forward = client.start_training(rnd + 1, receive_from_server=(c == 0 and rnd == 0))
                        for _ in range(self.hyper_params.client.local_epochs):
                            for _ in forward:
                                self.server.train_on_smashed_data(client.index)
                                client.backward()
                            client.end_epoch()
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
    def send_model(self, mbox="server") -> None:
        self.channel.send(Message(self.model, "client_model", self.index, inmemory=True), mbox)

    def receive_model(self, sender="server") -> None:
        msg = self.channel.receive(self.index, sender, msg_type="client_model")
        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def receive_client_info(self) -> int:
        msg = self.channel.receive(self.index, "server", msg_type="last_client_index")
        return msg.payload

    def start_training(self, current_round: int, receive_from_server=False) -> Generator:
        self.n_batches = 0
        self.running_loss = 0.0
        self.local_smashed = None
        self._load_from_cache()
        sender = "server" if receive_from_server else self.receive_client_info()
        self.receive_model(sender)
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        self.notify("start_fit", round=current_round, client_id=self.index, model=self.model)
        return self.forward_to_cut()

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

    def send_last_client_trained_info(self, client_index: int) -> None:
        self.channel.send(
            Message(self.last_client_trained_index, "last_client_index", "server", inmemory=True), client_index)

    def end_round(self, client_index):
        self.last_client_trained_index = client_index
        self.model.cpu()
        clear_cuda_cache()