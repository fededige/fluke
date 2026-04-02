import typing
from copy import deepcopy
from typing import Sequence, Generator, Iterable, Any

import torch

from fluke import FlukeENV, DDict
from fluke.client import Client
from fluke.comm import Message
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader, DataSplitter
from fluke.server import EarlyStopping
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict, aggregate_models
from playground.centralizedSL import CentralizedSL
from playground.clientSL import ClientSL
from playground.serverSL import ServerSL
from playground.splitfedv1 import ClientSplitFedV1


class ServerSplitFedV2(ServerSL):
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

    def broadcast_model(self, eligible: Sequence[ClientSplitFedV1]) -> None:
        self.channel.broadcast(
            Message(self.client_model, "client_model", "fed_server", inmemory=None), [c.index for c in eligible]
        )

    def receive_client_models(
            self, eligible: Sequence[Client], state_dict: bool = True
    ) -> Generator[torch.nn.Module, None, None]:
        for client in eligible:
            client_model = self.channel.receive("fed_server", client.index, "client_model").payload
            if state_dict:
                client_model = client_model.state_dict()
            yield client_model

    @torch.no_grad()
    def aggregate(
            self, eligible: Sequence[Client], client_models: Iterable[torch.nn.Module], result_model: torch.nn.Module
    ) -> None:
        weights = self._get_client_weights(eligible)
        aggregate_models(result_model, client_models, weights, self.hyper_params.lr, inplace=True)

    def end_client_round(self, client_index):
        self.model.cpu()
        clear_cuda_cache()

class SplitFedV2(CentralizedSL):
    def __init__(
            self,
            n_clients: int,
            data_splitter: DataSplitter,
            hyper_params: DDict | dict[str, Any],
            clients: list[ClientSplitFedV1] = None,
            server: ServerSplitFedV2 = None,
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
        return ClientSplitFedV1

    def get_server_class(self):
        return ServerSplitFedV2

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        self.server = typing.cast(ServerSplitFedV2, self.server)
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

                    eligible = typing.cast(Sequence[ClientSplitFedV1], self.server.get_eligible_clients(eligible_perc)) # eligible is still of type Sequence[Client] but the type checker knows that the return type is Sequence[ClientSplitFedV1]

                    self.server.broadcast_model(eligible) #il fed_server invia il modello client-side a tutti i client eligible (in questo caso server e fed_server sono coincidenti)

                    self.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                    for c, client in enumerate(eligible):
                        local_update = client.start_round(rnd + 1)
                        for _ in range(self.hyper_params.client.local_epochs):
                            for _ in local_update:
                                self.server.train_on_smashed_data(client.index)
                            self.server.end_epoch()

                        client.end_round(rnd + 1)
                        self.server.end_client_round(client.index)
                        progress_client.update(task_id=task_local, completed=c + 1)
                        progress_sl.update(task_id=task_rounds, advance=1)

                    client_models = self.server.receive_client_models(eligible, state_dict=False)
                    self.server.aggregate(eligible, client_models, self.server.client_model) #FedAvg over client models
                    # self.server.aggregate(eligible, [self.server.per_client_server_models[c.index] for c in eligible], self.server.model) #this aggregation is not needed in SFLV2
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