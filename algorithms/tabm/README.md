# TabM

**Purpose:** compare a neural ensemble with the linear and tree models.

**Status:** selected; dependency pinning, training and results are pending.

| Setting | Planned value |
| --- | --- |
| Tuned parameter | Blocks = 2, 3, 4 |
| Architecture | Width 128; 16 members; dropout 0.1; no numerical embeddings |
| Optimisation | AdamW; learning rate 0.0003; weight decay 0.0001; batch size 256; 100 epochs |
| Prediction identifier | `tabm` |

Load each training/validation partition from `processed_data/` through the
[shared input workflow](../README.md). Use all four A-D arms. Average member
losses per record, then apply the returned sample weights; average member
probabilities for prediction. Keep outer-test patients out of all training
decisions. Pin the model implementation and runtime dependencies before runs.
