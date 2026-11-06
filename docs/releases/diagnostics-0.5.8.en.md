# mindie-diagnostics 0.5.8: Validate the Windows scheduled GUI entry

[中文](diagnostics-0.5.8.md)

Windows service installation previously checked only that `pythonw.exe` existed. A console launcher bearing that name could still open a console window. Installation and explicit starts of existing registrations now read the PE32/PE32+ headers and require GUI subsystem 2. Missing, unreadable, malformed or console entries fail with `windowless_python_required` before registration or start.

Validation reads fixed-size headers without caching or modifying the interpreter. Repairing the environment permits a normal retry. Existing configuration, consent, virtual-environment identity and data-preservation behavior retain their current contracts.

Validation: the local Windows service installation, restart and network-context scope passed 71 tests with 6 platform-specific skips. The new regression covers rejection before installation, a replaced entry in an existing registration and recovery after repair. The formal release workflow separately runs the complete component suite and benchmarks the exact tagged source; those results ship as release assets.

Related audit: mindie-agent/knowledge#115.
