import numpy as np


def remove_model_data(self, target_container, target_mdl):
    getattr(self, target_container).pop(target_mdl)
    removed_mdl = self.dyn_mdls_dict[target_container].pop(target_mdl)
    self.dyn_mdls.pop(np.argmax(np.array(self.dyn_mdls, dtype=object) == removed_mdl))
    print(f"Model {target_container}: {target_mdl} was removed.")
