# mindie_sim hardware fixtures

Captured read-only from the standing clusters for Tier 0/1 parser tests.

- A2 `root@173.125.1.2`:
  - `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data/platform_config/Ascend910B4.ini`
  - `npu-smi info`, `npu-smi info -t common -i 0`, `npu-smi info -t board -i 0`
- A3 `root@192.168.13.165`:
  - `/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/data/platform_config/Ascend910_9391.ini`
  - `Ascend950DT_9581.ini` and `Ascend950DT_9582.ini` from the same CANN installation (A5 reference configuration, not a live A5 board)
  - `npu-smi info`, `npu-smi info -t common -i 0`, `npu-smi info -t board -i 0`

Board serial numbers are redacted. `cannsim_link_constants.py` is a small transcription of the
simulator ceiling values documented in `A5_HARDWARE_CONSTRAINTS.md`; it is not a bandwidth
measurement.
