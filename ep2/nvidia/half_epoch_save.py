from transformers import TrainerCallback
from swift.callbacks import callbacks_map

class HalfEpochSave(TrainerCallback):
    def __init__(self, args, trainer):
        self.epochs = args.num_train_epochs

    def on_train_begin(self, args, state, control, **kwargs):
        half_epochs = round(2 * self.epochs)
        self.milestones = {(state.max_steps * i + half_epochs - 1) // half_epochs
                           for i in range(1, half_epochs + 1)}
        if state.is_world_process_zero:
            print(f"Full checkpoints at half-epoch steps: {sorted(self.milestones)}", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.milestones:
            control.should_save = True
        return control

callbacks_map["pocket_half_epoch_save"] = HalfEpochSave
