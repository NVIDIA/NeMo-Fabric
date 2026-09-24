# Source notices

`hermes_security.py` derives from NVIDIA NemoClaw `test/hermes_native.py` at
revision `9146224da4`. On 2026-09-24 it was relocated to the Fabric-owned native
qualification suite without changing the upstream Apache-2.0 notices.

`qualify.py` is original Fabric test code. Both scripts run only against owned
local processes. The qualification script uses real installed harness code and
a local simulated inference endpoint; it does not prove external provider compatibility.
