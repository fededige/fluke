import typing
import torch

from fluke import FlukeENV
from fluke.comm import Message
from fluke.server import EarlyStopping
from fluke.utils import clear_cuda_cache
from playground.clientSL import ClientSL
from playground.serverSL import ServerSL
from playground.splitfedv2 import SplitFedV2

class ClientPSL(ClientSL):
    def start_round(self, current_round: int):
        self.n_batches = 0
        self.running_loss = 0.0
        self.local_smashed = None
        self._load_from_cache()
        if current_round == 1:
            self.receive_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

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

        self._check_persistency()
        self._save_to_cache()

class ServerPSL(ServerSL):

    def broadcast_model(self, eligible: typing.Sequence[ClientPSL]) -> None:
        self.channel.broadcast(
            Message(self.client_model, "client_model", "server", inmemory=None), [c.index for c in eligible]
        )

    def end_client_round(self, client_index):
        self.model.cpu()
        clear_cuda_cache()

class PSL(SplitFedV2):
    def get_client_class(self):
        return ClientPSL

    def get_server_class(self):
        return ServerPSL

    def _round_zero(self, eligible_perc) -> None:
        eligible = typing.cast(typing.Sequence[ClientPSL], self.server.get_eligible_clients(eligible_perc))
        self.server.broadcast_model(eligible)

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        self.server = typing.cast(ServerPSL, self.server)
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

                    for c, client in enumerate(eligible):
                        client.start_round(rnd + 1)
                        for _ in range(self.hyper_params.client.local_epochs):
                            local_update = client.train_epoch()
                            for _ in local_update:
                                self.server.train_on_smashed_data(client.index)
                            self.server.end_epoch()

                        client.end_round(rnd + 1)
                        self.server.end_client_round(client.index)
                        progress_client.update(task_id=task_local, completed=c + 1)
                        progress_sl.update(task_id=task_rounds, advance=1)

                    # we choose a random client for evaluation
                    random_client = eligible[0]
                    random_client.send_model()
                    self.server.receive_client_model(random_client.index)

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