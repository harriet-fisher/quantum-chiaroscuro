Subject: Jobs failing with engine_timeout and empty progress since 4 Oct (QDrive and tomography-api-v2)

Hi,

Since about 21:13 UTC on 4 Oct every job on my account has failed with `engine_timeout` ("retry the job", retryable: true), across two engines. Each failure is at about 62 s with an empty `progress` object, so the worker appears never to start.

Control cases, both tiny:
- tomography-api-v2, 2-qubit Bell circuit, one pair, 1024 shots, aer: job 88247923-3b74-46dd-982f-4599a03f54d7 (submitted 2026-10-05 13:50:35Z, failed 13:51:37Z).
- qdrive-api-v1, a 2-qubit Bell payload that completed in about 7 s earlier (job 27dd8027-7520-4e62-a855-963cda8a7541), resent unchanged: job e467d6c2-3b1d-4e0e-aa20-70eac2dbc6ea, failed at 62.7 s on 4 Oct.

Other failures: tomography-api-v2 cece6571-49da-48f8-91a6-644c573cf4c1 (18 qubits, 2026-10-05 13:46:18Z); qdrive-api-v1 c53afb2a-f67e-47fc-9d8b-d3b47c2abf9f and 30f5419b-5582-4f59-9a15-5873f4dcb932.

Questions:
1. Is there a known outage or degraded worker pool for these engines?
2. Are jobs that end in engine_timeout charged? I cannot find a balance endpoint, so I am comparing dashboard balances.
3. Is there a status page or health endpoint I should check before sending?

I am preparing a demo of a quantum-shaded projection piece and would like to run the tomography job once the service is healthy.

Thank you,
Harriet Fisher
