import stad.typehint as T
from stad import albu


class TrainerAugs:

    cfg: T.DictConfig

    def init_augs(self, data_type: str) -> T.Compose:

        augs = albu.load(self.cfg.augs[data_type].yaml, data_format="yaml")
        augs = self.update_augs(data_type, augs)
        self.save_augs(data_type, augs)
        return augs

    def update_augs(self, data_type: str, augs: T.Compose) -> T.Compose:

        for update in self.cfg.augs[data_type].updates:
            for i, aug in enumerate(augs):
                if aug.__class__.__name__ == update.name:
                    for k, v in update.args.items():
                        setattr(augs[i], k, v)
        return augs

    def save_augs(self, data_type: str, augs: T.Compose):

        albu.save(augs, f"hydra/{data_type}_augs.yaml", data_format="yaml")
