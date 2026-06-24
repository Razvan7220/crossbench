#!/usr/bin/env bash
#
# Generalized SWE-bench Pro smoke / mini-ablation dispatcher.
#
# Pass cells as ``--cell NAME:PROFILE_TAIL`` (repeatable). Profile tail
# is the schema-specific part — the base profiles (``online,haiku45_bedrock``
# by default) are prepended automatically. Pass ``--reruns N`` to run
# each cell N times with distinct run_ids; the aggregator prints
# mean ± std across reruns per cell at the end.
#
# Example: 2-cell smoke, single run each
# --------------------------------------
#   bash scripts/smoke_pro.sh \
#     --cell "nomem:no_memory" \
#     --cell "mem0v2:mem0v2_embedded,mem0_no_infer"
#
# Example: 2-cell smoke, 3 reruns
# -------------------------------
#   bash scripts/smoke_pro.sh --reruns 3 \
#     --cell "nomem:no_memory" \
#     --cell "mem0v2:mem0v2_embedded,mem0_no_infer"
#
# Flags
# -----
#   --cell NAME:PROFILE_TAIL   one cell (repeatable, required ≥1)
#   --reruns N                 reruns per cell (default 1)
#   --base-profiles CSV        prepended to every cell (default: online,haiku45_bedrock)
#   --task-id ID               override task list (repeatable; default = the 10
#                              stratified smoke instance_ids)
#   --sha SHA                  orchestrator image SHA (default: $(git rev-parse HEAD | cut -c1-7))
#   --build-id ID              CodeBuild id to wait for (default: skip wait)
#   --skip-wait                don't wait for any CodeBuild — assumes image is in ECR
#
# Cost / time
# -----------
# Per cell-rerun = 10 ECS tasks. 3 cells × 3 reruns = 9 dispatches = 90 ECS tasks.
# Wall-clock with full parallelism: ~30-60 min after orchestrator build finishes.
# Cost: ~$25 / 9 dispatches = ~$3 per cell-rerun on Haiku 4.5 via Bedrock.
#
# Output
# ------
# s3://<bucket>/runs/swebench_pro/pro-smoke-<NAME>-r<i>-<TS>/summary.json
# Per-cell mean ± std across reruns printed at the end.

set -euo pipefail

# Load .env from repo root (same as the cross-bench CLI does via python-dotenv)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$0")")"
[ -f "$REPO_ROOT/.env" ] && set -a && source "$REPO_ROOT/.env" && set +a

# --- Defaults -----------------------------------------------------------
RERUNS=1
BASE_PROFILES="online,haiku45_bedrock"
SHA="$(git rev-parse HEAD | cut -c1-7)"
BUILD_ID=""
S3_BUCKET="${S3_BUCKET:-cross-bench-dev-artifacts-3684f16}"
SKIP_WAIT=0
MAX_AGENT_STEPS=""  # empty = use benchmark TOML default (currently 250)
TASKS_PER_SHARD=1  # >1 reduces ECS concurrency (helps with Bedrock TPM limits)
BENCHMARK=""        # empty = auto (swebench_pro); pass swebench_pro_seal to use SEAL scaffold
STARTED_BY=""  # ECS "Started by" field; auto-derived from benchmark after arg parsing
TASK_SET=1    # 1 = original 20 (DEFAULT_TASK_IDS), 2 = second 31 (SECOND_RUN_TASK_IDS), 3 = random 15 (SET_3_TASK_IDS), 4 = random 50 (SET_4_TASK_IDS), 5 = random 100 (SET_5_TASK_IDS), 7 = 43 Alpine swerex-failing (SET_7_TASK_IDS), 8 = 11 Scale-only-resolved (SET_8_TASK_IDS)
LIMIT=""      # when set, grabs N from HuggingFace instead of curated task lists
OFFSET=""     # when set with LIMIT, takes ds[OFFSET:OFFSET+LIMIT] for chunked sweeps

CELLS=()  # array of "NAME:PROFILE_TAIL"
TASK_IDS=()
EXTRA_ENV=()  # forwarded as --extra-env KEY=VALUE to dispatch_swebench.py

# --- Default 20-task set — all gold-verified (2026-05-26) ---------------
# Every instance here resolves with the gold patch via swebench_pro_gold.
# See docs/swebench_pro/smoke_test_images.md for the full validation record.
DEFAULT_TASK_IDS=(
  "instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-a26c325bd8f6e2822d9d7e62f77a424c1db4fbf6-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_ansible__ansible-f327e65d11bb905ed9f15996024f857a95592629-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-395e5e20fab9cad517243372fa3c3c5d9e09ab2a-v7eee2454f617569fd6889f2211f75bc02a35f9f8"
  "instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a"
  "instance_gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-3fa6904377c006497169945428e8197158667910-v626ec2a48416b10a88641359a169d99e935ff037"
  "instance_gravitational__teleport-6eaaf3a27e64f4ef4ef855bd35d7ec338cf17460-v626ec2a48416b10a88641359a169d99e935ff037"
  "instance_gravitational__teleport-c782838c3a174fdff80cafd8cd3b1aa4dae8beb2"
  "instance_internetarchive__openlibrary-f343c08f89c772f7ba6c0246f384b9e6c3dc0add-v08d8e8889ec945ab821fb156c04c7d2e2810debb"
  "instance_internetarchive__openlibrary-25858f9f0c165df25742acf8309ce909773f0cdd-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-4a5d2a7d24c9e4c11d3069220c0685b736d5ecde-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-dbbd9d539c6d4fd45d5be9662aa19b6d664b5137-v08d8e8889ec945ab821fb156c04c7d2e2810debb"
  "instance_internetarchive__openlibrary-8a5a63af6e0be406aa6c8c9b6d5f28b2f1b6af5a-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_navidrome__navidrome-1e96b858a91c640fe64e84c5e5ad8cc0954ea38d"
  "instance_navidrome__navidrome-7073d18b54da7e53274d11c9e2baef1242e8769e"
  "instance_navidrome__navidrome-b65e76293a917ee2dfc5d4b373b1c62e054d0dca"
  "instance_NodeBB__NodeBB-51d8f3b195bddb13a13ddc0de110722774d9bb1b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_qutebrowser__qutebrowser-cc360cd4a34a126274c7b51f3b63afbaf3e05a02-v5fc38aaf22415ab0b70567368332beee7955b367"
)

# --- Second run — 30 candidates (instances 21-50), pending gold-patch validation ---
# Validate with: bash scripts/smoke_pro.sh --benchmark swebench_pro_gold --task-set 2 --cell "gold:no_memory"
SECOND_RUN_TASK_IDS=(
  "instance_qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-v059c6fdc75567943479b23ebca7c07b5e9a7f34c"
  "instance_qutebrowser__qutebrowser-c580ebf0801e5a3ecabc54f327498bb753c6d5f2-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-f631cd4422744160d9dcf7a0455da532ce973315-v35616345bb8052ea303186706cec663146f0f184"
  "instance_qutebrowser__qutebrowser-96b997802e942937e81d2b8a32d08f00d3f4bc4e-v5fc38aaf22415ab0b70567368332beee7955b367"
  "instance_element-hq__element-web-33e8edb3d508d6eefb354819ca693b7accc695e7"
  "instance_future-architect__vuls-407407d306e9431d6aa0ab566baa6e44e5ba2904"
  "instance_internetarchive__openlibrary-111347e9583372e8ef91c82e0612ea437ae3a9c9-v2d9a6c849c60ed19fd0858ce9e40b7cc8e097e59"
  "instance_future-architect__vuls-e6c0da61324a0c04026ffd1c031436ee2be9503a"
  "instance_qutebrowser__qutebrowser-fd6790fe8c02b144ab2464f1fc8ab3d02ce3c476-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_NodeBB__NodeBB-397835a05a8e2897324e566b41c5e616e172b4af-v89631a1cdb318276acb48860c5d78077211397c6"
  "instance_flipt-io__flipt-e42da21a07a5ae35835ec54f74004ebd58713874"
  "instance_flipt-io__flipt-3b2c25ee8a3ac247c3fad13ad8d64ace34ec8ee7"
  "instance_NodeBB__NodeBB-a5afad27e52fd336163063ba40dcadc80233ae10-vd59a5728dfc977f44533186ace531248c2917516"
  "instance_gravitational__teleport-eefac60a350930e5f295f94a2d55b94c1988c04e-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_NodeBB__NodeBB-f9ce92df988db7c1ae55d9ef96d247d27478bc70-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08"
  "instance_future-architect__vuls-2c84be80b65d022c262956cd26fc79d8bb2f7010"
  "instance_flipt-io__flipt-c12967bc73fdf02054cf3ef8498c05e25f0a18c0"
  "instance_qutebrowser__qutebrowser-0fc6d1109d041c69a68a896db87cf1b8c194cef7-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_gravitational__teleport-0ac7334939981cf85b9591ac295c3816954e287e"
  "instance_gravitational__teleport-b5d8169fc0a5e43fee2616c905c6d32164654dc6"
  "instance_future-architect__vuls-86b60e1478e44d28b1aff6b9ac7e95ceb05bc5fc"
  "instance_qutebrowser__qutebrowser-70248f256f93ed9b1984494d0a1a919ddd774892-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_ansible__ansible-e40889e7112ae00a21a2c74312b330e67a766cc0-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_navidrome__navidrome-27875ba2dd1673ddf8affca526b0664c12c3b98b"
  "instance_internetarchive__openlibrary-11838fad1028672eb975c79d8984f03348500173-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  # Run 3 — 3 replacements for the removed protonmail/Alpine instance (#51-53)
  # Gold-validated 2026-05-27 (pro-smoke-gold3-r1-20260527-183851), all 3 pass.
  "instance_ansible__ansible-5e369604e1930b1a2e071fecd7ec5276ebd12cb1-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_internetarchive__openlibrary-b4f7c185ae5f1824ac7f3a18e8adf6a4b468459c-v08d8e8889ec945ab821fb156c04c7d2e2810debb"
  "instance_gravitational__teleport-3ff75e29fb2153a2637fe7f83e49dc04b1c99c9f"
)

# --- Set 3 — 15 random instances from the full 731-row Pro benchmark -----
# Sampled 2026-06-05 with random.seed(42); 4 known-bad collisions swapped via
# seeds 43/44 against the known-failing repo/suffix patterns. Status unverified.
# Covers all four benchmark languages (Python, JavaScript, TypeScript, Go).
# See docs/swebench_pro/smoke_test_images.md § "Set 3" for the full record.
SET_3_TASK_IDS=(
  "instance_qutebrowser__qutebrowser-7f9713b20f623fc40473b7167a082d6db0f0fd40-va0fd88aac89cde702ec1ba84877234da33adce8a"
  "instance_tutao__tutanota-f3ffe17af6e8ab007e8d461355057ad237846d9d-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf"
  "instance_ansible__ansible-83909bfa22573777e3db5688773bda59721962ad-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_gravitational__teleport-b4e7cd3a5e246736d3fe8d6886af55030b232277"
  "instance_navidrome__navidrome-23bebe4e06124becf1000e88472ae71a6ca7de4c"
  "instance_flipt-io__flipt-7161f7b876773a911afdd804b281e52681cb7321"
  "instance_internetarchive__openlibrary-43f9e7e0d56a4f1d487533543c17040a029ac501-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_ansible__ansible-5f4e332e3762999d94af27746db29ff1729252c1-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_qutebrowser__qutebrowser-479aa075ac79dc975e2e949e188a328e95bf78ff-vc2f56a753b62a190ddb23cd330c257b9cf560d12"
  "instance_internetarchive__openlibrary-d8162c226a9d576f094dc1830c4c1ffd0be2dd17-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c"
  "instance_future-architect__vuls-e049df50fa1eecdccc5348e27845b5c783ed7c76-v73dc95f6b90883d8a87e01e5e9bb6d3cc32add6d"
  "instance_flipt-io__flipt-507170da0f7f4da330f6732bffdf11c4df7fc192"
  "instance_flipt-io__flipt-a0cbc0cb65ae601270bdbe3f5313e2dfd49c80e4"
  "instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08"
  "instance_NodeBB__NodeBB-9c576a0758690f45a6ca03b5884c601e473bf2c1-vd59a5728dfc977f44533186ace531248c2917516"
)

# --- Set 4 — 50 random instances from the full 731-row Pro benchmark -----
# Sampled 2026-06-08 with random.seed(20260608); excludes Sets 1/2/3, known-
# failing IDs, and the entire protonmail/webclients repo (Alpine/musl,
# deferred per workflow.md S23). Status unverified — validate via
# `--benchmark swebench_pro_gold --task-set 4` before treating as smoke
# targets. See docs/ablation/set_4.md for the full record + reproducibility
# recipe.
SET_4_TASK_IDS=(
  "instance_NodeBB__NodeBB-0c81642997ea1d827dbd02c311db9d4976112cd4-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-2657804c1fb6b84dc76ad3b18ecf061aaab5f29f-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-445b70deda20201b7d9a68f7224da751b3db728c-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed"
  "instance_NodeBB__NodeBB-8ca65b0c78c67c1653487c02d1135e1b702185e1-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-bad15643013ca15affe408b75eba9e47cc604bb2-vd59a5728dfc977f44533186ace531248c2917516"
  "instance_ansible__ansible-185d41031660a676c43fbb781cd1335902024bfe-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-1a4644ff15355fd696ac5b9d074a566a80fe7ca3-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
  "instance_ansible__ansible-379058e10f3dbc0fdcaf80394bd09b18927e7d33-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-3889ddeb4b780ab4bac9ca2e75f8c1991bcabe83-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_ansible__ansible-39bd8b99ec8c6624207bf3556ac7f9626dad9173-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-3db08adbb1cc6aa9941be5e0fc810132c6e1fa4b-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-9759e0ca494de1fd5fc2df2c5d11c57adbe6007c-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-c1f2df47538b884a43320f53e787197793b105e8-v906c969b551b346ef54a2c0b41e04f632b7b73c2"
  "instance_ansible__ansible-d30fc6c0b359f631130b0e979d9a78a7b3747d48-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-d58e69c82d7edd0583dd8e78d76b075c33c3151e-v173091e2e36d38c978002990795f66cfc0af30ad"
  "instance_ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_flipt-io__flipt-0b119520afca1cf25c470ff4288c464d4510b944"
  "instance_flipt-io__flipt-3d5a345f94c2adc8a0eaa102c189c08ad4c0f8e8"
  "instance_flipt-io__flipt-c1fd7a81ef9f23e742501bfb26d914eb683262aa"
  "instance_flipt-io__flipt-e2bd19dafa7166c96b082fb2a59eb54b4be0d778"
  "instance_flipt-io__flipt-ea9a2663b176da329b3f574da2ce2a664fc5b4a1"
  "instance_future-architect__vuls-139f3a81b66c47e6d8f70ce6c4afe7a9196a6ea8"
  "instance_future-architect__vuls-78b52d6a7f480bd610b692de9bf0c86f57332f23"
  "instance_future-architect__vuls-878c25bf5a9c9fd88ac32eb843f5636834d5712d"
  "instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07"
  "instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08b182eca4e2b9a17-v264a82e2f4818e30f5a25e4da53b27ba119f62b5"
  "instance_future-architect__vuls-fd18df1dd4e4360f8932bc4b894bd8b40d654e7c"
  "instance_gravitational__teleport-10123c046e21e1826098e485a4c2212865a49d9f"
  "instance_gravitational__teleport-87a593518b6ce94624f6c28516ce38cc30cbea5a"
  "instance_gravitational__teleport-a95b3ae0667f9e4b2404bf61f51113e6d83f01cd"
  "instance_gravitational__teleport-baeb2697c4e4870c9850ff0cd5c7a2d08e1401c9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-d873ea4fa67d3132eccba39213c1ca2f52064dcc-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_internetarchive__openlibrary-0a90f9f0256e4f933523e9842799e39f95ae29ce-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c"
  "instance_internetarchive__openlibrary-3f580a5f244c299d936d73d9e327ba873b6401d9-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-427f1f4eddfc54735ca451779d4f95bf683d1b0e-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-4b7ea2977be2747496ba792a678940baa985f7ea-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-62d2243131a9c7e6aee00d1e9c5660fd5b594e89-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-798a582540019363d14b2090755cc7b89a350788-v430f20c722405e462d9ef44dee7d34c41e76fe7a"
  "instance_internetarchive__openlibrary-910b08570210509f3bcfebf35c093a48243fe754-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-9cd47f4dc21e273320d9e30d889c864f8cb20ccf-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-bdba0af0f6cbaca8b5fc3be2a3080f38156d9c92-ve8c8d62a2b60610a3c4631f5f23ed866bada9818"
  "instance_internetarchive__openlibrary-c4eebe6677acc4629cb541a98d5e91311444f5d4-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-fad4a40acf5ff5f06cd7441a5c7baf41a7d81fe4-vfa6ff903cb27f336e17654595dd900fa943dcd91"
  "instance_navidrome__navidrome-29bc17acd71596ae92131aca728716baf5af9906"
  "instance_navidrome__navidrome-5001518260732e36d9a42fb8d4c054b28afab310"
  "instance_navidrome__navidrome-fa85e2a7816a6fe3829a4c0d8e893e982b0985da"
  "instance_qutebrowser__qutebrowser-473a15f7908f2bb6d670b0e908ab34a28d8cf7e2-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
  "instance_qutebrowser__qutebrowser-ef5ba1a0360b39f9eff027fbdc57f363597c3c3b-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
  "instance_tutao__tutanota-8513a9e8114a8b42e64f4348335e0f23efa054c4-vee878bb72091875e912c52fc32bc60ec3760227b"
  "instance_tutao__tutanota-fe240cbf7f0fdd6744ef7bef8cb61676bcdbb621-vc4e41fd0029957297843cb9dec4a25c7c756f029"
)

# --- Set 5 — 100 random instances from the full 731-row Pro benchmark ----
# Sampled 2026-06-11 with random.seed(20260611); excludes Sets 1/2/3/4, known-
# failing IDs, and the entire protonmail/webclients repo (Alpine/musl,
# deferred per workflow.md S23). Status unverified — validate via
# `--benchmark swebench_pro_gold --task-set 5` before treating as smoke
# targets. See docs/ablation/set_5.md for the full record + reproducibility
# recipe.
SET_5_TASK_IDS=(
  "instance_NodeBB__NodeBB-087e6020e490b4a1759f38c1ad03869511928263-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-18c45b44613aecd53e9f60457b9812049ab2998d-v0495b863a912fbff5749c67e860612b91825407c"
  "instance_NodeBB__NodeBB-6489e9fd9ed16ea743cc5627f4d86c72fbdb3a8a-v2c59007b1005cd5cd14cbb523ca5229db1fd2dd8"
  "instance_NodeBB__NodeBB-76c6e30282906ac664f2c9278fc90999b27b1f48-vd59a5728dfc977f44533186ace531248c2917516"
  "instance_NodeBB__NodeBB-8168c6c40707478f71b8af60300830fe554c778c-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-be43cd25974681c9743d424238b7536c357dc8d3-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_NodeBB__NodeBB-f48ed3658aab7be0f1165d4c1f89af48d7865189-v0495b863a912fbff5749c67e860612b91825407c"
  "instance_ansible__ansible-0fd88717c953b92ed8a50495d55e630eb5d59166-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-189fcb37f973f0b1d52b555728208eeb9a6fce83-v906c969b551b346ef54a2c0b41e04f632b7b73c2"
  "instance_ansible__ansible-1b70260d5aa2f6c9782fd2b848e8d16566e50d85-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-1ee70fc272aff6bf3415357c6e13c5de5b928d9b-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-34db57a47f875d11c4068567b9ec7ace174ec4cf-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_ansible__ansible-3b823d908e8a5d17674f8c26d337d3114b7493b1-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_ansible__ansible-40ade1f84b8bb10a63576b0ac320c13f57c87d34-v6382ea168a93d80a64aab1fbd8c4f02dc5ada5bf"
  "instance_ansible__ansible-622a493ae03bd5e5cf517d336fc426e9d12208c7-v906c969b551b346ef54a2c0b41e04f632b7b73c2"
  "instance_ansible__ansible-9142be2f6cabbe6597c9254c5bb9186d17036d55-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_ansible__ansible-b5e0293645570f3f404ad1dbbe5f006956ada0df-v0f01c69f1e2528b935359cfe578530722bca2c59"
  "instance_ansible__ansible-b6290e1d156af608bd79118d209a64a051c55001-v390e508d27db7a51eece36bb6d9698b63a5b638a"
  "instance_ansible__ansible-bec27fb4c0a40c5f8bbcf26a475704227d65ee73-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
  "instance_ansible__ansible-e0c91af45fa9af575d10fd3e724ebc59d2b2d6ac-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
  "instance_ansible__ansible-f02a62db509dc7463fab642c9c3458b9bc3476cc-v390e508d27db7a51eece36bb6d9698b63a5b638a"
  "instance_ansible__ansible-f8ef34672b961a95ec7282643679492862c688ec-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_element-hq__element-web-f14374a51c153f64f313243f2df6ea4971db4e15"
  "instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1"
  "instance_flipt-io__flipt-15b76cada1ef29cfa56b0fba36754be36243dded"
  "instance_flipt-io__flipt-36e62baffae2132f78f9d34dc300a9baa2d7ae0e"
  "instance_flipt-io__flipt-5aef5a14890aa145c22d864a834694bae3a6f112"
  "instance_flipt-io__flipt-65581fef4aa807540cb933753d085feb0d7e736f"
  "instance_flipt-io__flipt-6fd0f9e2587f14ac1fdd1c229f0bcae0468c8daa"
  "instance_flipt-io__flipt-72d06db14d58692bfb4d07b1aa745a37b35956f3"
  "instance_flipt-io__flipt-96820c3ad10b0b2305e8877b6b303f7fafdf815f"
  "instance_flipt-io__flipt-9f8127f225a86245fa35dca4885c2daef824ee55"
  "instance_flipt-io__flipt-b6cef5cdc0daff3ee99e5974ed60a1dc6b4b0d67"
  "instance_flipt-io__flipt-cd18e54a0371fa222304742c6312e9ac37ea86c1"
  "instance_flipt-io__flipt-e50808c03e4b9d25a6a78af9c61a3b1616ea356b"
  "instance_flipt-io__flipt-e88e93990e3ec1e7697754b423decc510d5dd5fe"
  "instance_future-architect__vuls-0ec945d0510cdebf92cdd8999f94610772689f14"
  "instance_future-architect__vuls-457a3a9627fb9a0800d0aecf1d4713fb634a9011"
  "instance_future-architect__vuls-4a72295de7b91faa59d90a5bee91535bbe76755d"
  "instance_future-architect__vuls-4c04acbd9ea5b073efe999e33381fa9f399d6f27"
  "instance_future-architect__vuls-61c39637f2f3809e1b5dad05f0c57c799dce1587"
  "instance_future-architect__vuls-6eff6a9329a65cc412e79b8f82444dfa3d0f0b5a"
  "instance_future-architect__vuls-83bcca6e669ba2e4102f26c4a2b52f78c7861f1a"
  "instance_future-architect__vuls-8d5ea98e50cf616847f4e5a2df300395d1f719e9"
  "instance_future-architect__vuls-a76302c11174ca081f656c63a000ffa746e350af"
  "instance_future-architect__vuls-aaea15e516ece43978cf98e09e52080478b1d39f"
  "instance_future-architect__vuls-d18e7a751d07260d75ce3ba0cd67c4a6aebfd967"
  "instance_future-architect__vuls-e1fab805afcfc92a2a615371d0ec1e667503c254-v264a82e2f4818e30f5a25e4da53b27ba119f62b5"
  "instance_future-architect__vuls-f6509a537660ea2bce0e57958db762edd3a36702"
  "instance_future-architect__vuls-fe8d252c51114e922e6836055ef86a15f79ad042"
  "instance_gravitational__teleport-1330415d33a27594c948a36d9d7701f496229e9f"
  "instance_gravitational__teleport-1b08e7d0dbe68fe530a0f08ad408ec198b7c53fc-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-288c5519ce0dec9622361a5e5d6cd36aa2d9e348"
  "instance_gravitational__teleport-37c3724d0d6637e959e39408ee351565d73afe71-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-46a13210519461c7cec8d643bfbe750265775b41"
  "instance_gravitational__teleport-af5e2517de7d18406b614e413aca61c319312171-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-dd3977957a67bedaf604ad6ca255ba8c7b6704e9"
  "instance_gravitational__teleport-e6681abe6a7113cfd2da507f05581b7bdf398540-v626ec2a48416b10a88641359a169d99e935ff037"
  "instance_internetarchive__openlibrary-00bec1e7c8f3272c469a58e1377df03f955ed478-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-1894cb48d6e7fb498295a5d3ed0596f6f603b784-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-7bf3238533070f2d24bafbb26eedf675d51941f6-v08d8e8889ec945ab821fb156c04c7d2e2810debb"
  "instance_internetarchive__openlibrary-7edd1ef09d91fe0b435707633c5cc9af41dedddf-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c"
  "instance_internetarchive__openlibrary-a48fd6ba9482c527602bc081491d9e8ae6e8226c-vfa6ff903cb27f336e17654595dd900fa943dcd91"
  "instance_internetarchive__openlibrary-b112069e31e0553b2d374abb5f9c5e05e8f3dbbe-ve8c8d62a2b60610a3c4631f5f23ed866bada9818"
  "instance_internetarchive__openlibrary-c12943be1db80cf1114bc267ddf4f9933aca9b28-v2c55207218fb8a0138425cbf7d9675272e240b90"
  "instance_internetarchive__openlibrary-c8996ecc40803b9155935fd7ff3b8e7be6c1437c-ve8fc82d8aae8463b752a211156c5b7b59f349237"
  "instance_internetarchive__openlibrary-d40ec88713dc95ea791b252f92d2f7b75e107440-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-f0341c0ba81c790241b782f5103ce5c9a6edf8e3-ve8fc82d8aae8463b752a211156c5b7b59f349237"
  "instance_internetarchive__openlibrary-f3b26c2c0721f8713353fe4b341230332e30008d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-fdbc0d8f418333c7e575c40b661b582c301ef7ac-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_navidrome__navidrome-5e549255201e622c911621a7b770477b1f5a89be"
  "instance_navidrome__navidrome-669c8f4c49a7ef51ac9a53c725097943f67219eb"
  "instance_navidrome__navidrome-6b3b4d83ffcf273b01985709c8bc5df12bbb8286"
  "instance_navidrome__navidrome-89b12b34bea5687c70e4de2109fd1e7330bb2ba2"
  "instance_navidrome__navidrome-8d56ec898e776e7e53e352cb9b25677975787ffc"
  "instance_navidrome__navidrome-c90468b895f6171e33e937ff20dc915c995274f0"
  "instance_navidrome__navidrome-f7d4fcdcc1a59d1b4f835519efb402897757e371"
  "instance_qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-233cb1cc48635130e5602549856a6fa4ab4c087f-v35616345bb8052ea303186706cec663146f0f184"
  "instance_qutebrowser__qutebrowser-305e7c96d5e2fdb3b248b27dfb21042fb2b7e0b8-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-36ade4bba504eb96f05d32ceab9972df7eb17bcc-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-3e21c8214a998cb1058defd15aabb24617a76402-v5fc38aaf22415ab0b70567368332beee7955b367"
  "instance_qutebrowser__qutebrowser-5cef49ff3074f9eab1da6937a141a39a20828502-v02ad04386d5238fe2d1a1be450df257370de4b6a"
  "instance_qutebrowser__qutebrowser-77c3557995704a683cdb67e2a3055f7547fa22c3-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
  "instance_qutebrowser__qutebrowser-7b603dd6bf195e3e723ce08ff64a82b406e3f6b6-v9f8e9d96c85c85a605e382f1510bd08563afc566"
  "instance_qutebrowser__qutebrowser-8f46ba3f6dc7b18375f7aa63c48a1fe461190430-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-996487c43e4fcc265b541f9eca1e7930e3c5cf05-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
  "instance_qutebrowser__qutebrowser-9b71c1ea67a9e7eb70dd83214d881c2031db6541-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"
  "instance_qutebrowser__qutebrowser-e70f5b03187bdd40e8bf70f5f3ead840f52d1f42-v02ad04386d5238fe2d1a1be450df257370de4b6a"
  "instance_qutebrowser__qutebrowser-ebfe9b7aa0c4ba9d451f993e08955004aaec4345-v059c6fdc75567943479b23ebca7c07b5e9a7f34c"
  "instance_qutebrowser__qutebrowser-ec2dcfce9eee9f808efc17a1b99e227fc4421dea-v5149fcda2a9a6fe1d35dfed1bade1444a11ef271"
  "instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c"
  "instance_qutebrowser__qutebrowser-f8e7fea0becae25ae20606f1422068137189fe9e"
  "instance_tutao__tutanota-09c2776c0fce3db5c6e18da92b5a45dce9f013aa-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf"
  "instance_tutao__tutanota-12a6cbaa4f8b43c2f85caca0787ab55501539955-vc4e41fd0029957297843cb9dec4a25c7c756f029"
  "instance_tutao__tutanota-219bc8f05d7b980e038bc1524cb021bf56397a1b-vee878bb72091875e912c52fc32bc60ec3760227b"
  "instance_tutao__tutanota-40e94dee2bcec2b63f362da283123e9df1874cc1-vc4e41fd0029957297843cb9dec4a25c7c756f029"
  "instance_tutao__tutanota-51818218c6ae33de00cbea3a4d30daac8c34142e-vc4e41fd0029957297843cb9dec4a25c7c756f029"
  "instance_tutao__tutanota-d1aa0ecec288bfc800cfb9133b087c4f81ad8b38-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf"
  "instance_tutao__tutanota-db90ac26ab78addf72a8efaff3c7acc0fbd6d000-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf"
)

# --- Set 7 — 43 Alpine-only swerex-failing instances ---------------------
# Sourced from docs/swebench_pro/swerex_failing_intersect.txt — the task
# IDs that reliably failed swerex bring-up across every prior SEAL run
# before the Sub #8 alpine_dockerfile fix landed (see workflow.md S23
# "patches 1-5 + #8"). Every task here is in one of the Alpine/musl Pro
# repos (flipt-io/flipt, future-architect/vuls, gravitational/teleport,
# navidrome/navidrome). Purpose: validate the Alpine fix's impact on the
# FULL Alpine population — Set 4 only had 16/50 Alpine tasks at
# n=50 the per-repo signal is sample-limited; Set 7 gives 43 Alpine tasks
# in one shot for a cleaner before/after read on the headline pass rate.
SET_7_TASK_IDS=(
  "instance_flipt-io__flipt-86906cbfc3a5d3629a583f98e6301142f5f14bdb-v6bea0cc3a6fc532d7da914314f2944fc1cd04dee"
  "instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07"
  "instance_future-architect__vuls-e049df50fa1eecdccc5348e27845b5c783ed7c76-v73dc95f6b90883d8a87e01e5e9bb6d3cc32add6d"
  "instance_future-architect__vuls-e1fab805afcfc92a2a615371d0ec1e667503c254-v264a82e2f4818e30f5a25e4da53b27ba119f62b5"
  "instance_future-architect__vuls-e4728e388120b311c4ed469e4f942e0347a2689b-v264a82e2f4818e30f5a25e4da53b27ba119f62b5"
  "instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08b182eca4e2b9a17-v264a82e2f4818e30f5a25e4da53b27ba119f62b5"
  "instance_gravitational__teleport-005dcb16bacc6a5d5890c4cd302ccfd4298e275d-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08edd8b84a9511b4-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-0ecf31de0e98b272a6a2610abe1bbedd379a38a3-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-1a77b7945a022ab86858029d30ac7ad0d5239d00-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-1b08e7d0dbe68fe530a0f08ad408ec198b7c53fc-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-24cafecd8721891092210afc55f6413ab46ca211-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-2b15263e49da5625922581569834eec4838a9257-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-2bb3bbbd8aff1164a2353381cb79e1dc93b90d28-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-32bcd71591c234f0d8b091ec01f1f5cbfdc0f13c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-37c3724d0d6637e959e39408ee351565d73afe71-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-3a5c1e26394df2cb4fb3f01147fb9979662972c5-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-3ff19cf7c41f396ae468797d3aeb61515517edc9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-46aa81b1ce96ebb4ebed2ae53fd78cd44a05da6c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-47530e1fd8bfb84ec096ebcbbc29990f30829655-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-4e1c39639edf1ab494dd7562844c8b277b5cfa18-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-4f771403dc4177dc26ee0370f7332f3fe54bee0f-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-53814a2d600ccd74c1e9810a567563432b98386e-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-645afa051b65d137654fd0d2d878a700152b305a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-65438e6e44b6ce51458d09b7bb028a2797cfb0ea-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-73cc189b0e9636d418c4470ecce0d9af5dae2f02-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-7744f72c6eb631791434b648ba41083b5f6d2278-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-8302d467d160f869b77184e262adbe2fbc95d9ba-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-ad41b3c15414b28a6cec8c25424a19bfa7abd0e9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-af5e2517de7d18406b614e413aca61c319312171-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-b1bcd8b90c474a35bb11cc3ef4cc8941e1f8eab2-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-b8fbb2d1e90ffcde88ed5fe9920015c1be075788-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-baeb2697c4e4870c9850ff0cd5c7a2d08e1401c9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-bb562408da4adeae16e025be65e170959d1ec492-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-cb712e3f0b06dadc679f895daef8072cae400c26-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-d6ffe82aaf2af1057b69c61bf9df777f5ab5635a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_gravitational__teleport-d873ea4fa67d3132eccba39213c1ca2f52064dcc-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-e6681abe6a7113cfd2da507f05581b7bdf398540-v626ec2a48416b10a88641359a169d99e935ff037"
  "instance_gravitational__teleport-e6895d8934f6e484341034869901145fbc025e72-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_gravitational__teleport-eefac60a350930e5f295f94a2d55b94c1988c04e-vee9b09fb20c43af7e520f57e9239bbcf46b7113d"
  "instance_navidrome__navidrome-66b74c81f115c78cb69910b0472eeb376750efc4"
)

# ============================================================================
# SET_8 — "Scale-resolved-only" diagnostic set for SEAL Haiku 4.5 reproduction.
# ============================================================================
#
# These are the 11 instances where Scale's Oct 2025 Haiku 4.5 run resolved
# the task but CrossBench's same-config run did not (verified on the
# `pro-smoke-direct_temp1_no_memory-r1-20260615-212633` 50-shard run on
# Set 4 — see docs/ablation/set_8.md for the full audit).
#
# All 11 ended in CrossBench with ``exit_status=Submitted``, ``returncode=0``,
# ``swerex_infra_error=None`` — i.e. model finished cleanly, applied a
# patch, but the patch failed grading. Scale produced a correct patch on
# the same task. The gap is purely model-output divergence.
#
# Use this set with ``--reruns 3`` and a tracing-enabled profile to capture
# the full sweagent .traj + LiteLLM I/O for side-by-side comparison against
# Scale's published trajectories at:
#   s3://scaleapi-results/swe-bench-pro/claude-45haiku-10222025/traj/<id>/<id>.traj
#
# Repo distribution:
#   internetarchive/openlibrary  6 (Python Flask)
#   ansible/ansible              3 (Python collections/galaxy)
#   flipt-io/flipt               1 (Go, simple auth tweak)
#   navidrome/navidrome          1 (Go, ttlcache wrap)
SET_8_TASK_IDS=(
  "instance_ansible__ansible-185d41031660a676c43fbb781cd1335902024bfe-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_ansible__ansible-1a4644ff15355fd696ac5b9d074a566a80fe7ca3-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
  "instance_ansible__ansible-3db08adbb1cc6aa9941be5e0fc810132c6e1fa4b-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
  "instance_flipt-io__flipt-0b119520afca1cf25c470ff4288c464d4510b944"
  "instance_internetarchive__openlibrary-3f580a5f244c299d936d73d9e327ba873b6401d9-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-427f1f4eddfc54735ca451779d4f95bf683d1b0e-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-62d2243131a9c7e6aee00d1e9c5660fd5b594e89-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4"
  "instance_internetarchive__openlibrary-bdba0af0f6cbaca8b5fc3be2a3080f38156d9c92-ve8c8d62a2b60610a3c4631f5f23ed866bada9818"
  "instance_internetarchive__openlibrary-c4eebe6677acc4629cb541a98d5e91311444f5d4-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_internetarchive__openlibrary-fad4a40acf5ff5f06cd7441a5c7baf41a7d81fe4-vfa6ff903cb27f336e17654595dd900fa943dcd91"
  "instance_navidrome__navidrome-29bc17acd71596ae92131aca728716baf5af9906"
)


# --- Argparse ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cell) CELLS+=("$2"); shift 2 ;;
    --reruns) RERUNS="$2"; shift 2 ;;
    --base-profiles) BASE_PROFILES="$2"; shift 2 ;;
    --task-id) TASK_IDS+=("$2"); shift 2 ;;
    --sha) SHA="$2"; shift 2 ;;
    --build-id) BUILD_ID="$2"; shift 2 ;;
    --skip-wait) SKIP_WAIT=1; shift ;;
    --max-agent-steps) MAX_AGENT_STEPS="$2"; shift 2 ;;
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --started-by) STARTED_BY="$2"; shift 2 ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    -n|--limit) LIMIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --tasks-per-shard) TASKS_PER_SHARD="$2"; shift 2 ;;
    --extra-env) EXTRA_ENV+=("$2"); shift 2 ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | head -n -2
      exit 0
      ;;
    *) echo "Unknown flag: $1"; exit 2 ;;
  esac
done

if [ "${#CELLS[@]}" -eq 0 ]; then
  echo "ERROR: at least one --cell is required" >&2
  echo "Run --help for usage." >&2
  exit 2
fi

# --- Auto-forward Slack env vars from shell/.env to the ECS container ----
# Once SLACK_WEBHOOK_URL and CROSSBENCH_SLACK_PER_TASK are set in .env (or
# exported in the shell), every dispatch picks them up automatically —
# no need to repeat --extra-env on every invocation. An explicit
# --extra-env always wins (we skip when the same key is already present).
_has_extra() {
  local needle="$1"
  for kv in "${EXTRA_ENV[@]}"; do
    [[ "$kv" == "${needle}="* ]] && return 0
  done
  return 1
}
for var in CROSSBENCH_SLACK_PER_TASK SLACK_WEBHOOK_URL; do
  if [ -n "${!var:-}" ] && ! _has_extra "$var"; then
    EXTRA_ENV+=("${var}=${!var}")
  fi
done

INDEX_OFFSET=0
if [ "${#TASK_IDS[@]}" -eq 0 ]; then
  if [ "$TASK_SET" = "2" ]; then
    TASK_IDS=("${SECOND_RUN_TASK_IDS[@]}")
    INDEX_OFFSET="${#DEFAULT_TASK_IDS[@]}"   # labels start after run-1
  elif [ "$TASK_SET" = "3" ]; then
    TASK_IDS=("${SET_3_TASK_IDS[@]}")
    INDEX_OFFSET=53                           # labels start at 54 to match docs § "Set 3"
  elif [ "$TASK_SET" = "4" ]; then
    TASK_IDS=("${SET_4_TASK_IDS[@]}")
    INDEX_OFFSET=68                           # labels start at 69; Set 3 ended at 68 (54-68 = 15 IDs)
  elif [ "$TASK_SET" = "5" ]; then
    TASK_IDS=("${SET_5_TASK_IDS[@]}")
    INDEX_OFFSET=118                          # labels start at 119; Set 4 ended at 118 (69-118 = 50 IDs)
  elif [ "$TASK_SET" = "7" ]; then
    TASK_IDS=("${SET_7_TASK_IDS[@]}")
    INDEX_OFFSET=218                          # labels start at 219; Set 5 ended at 218 (119-218 = 100 IDs)
  elif [ "$TASK_SET" = "8" ]; then
    TASK_IDS=("${SET_8_TASK_IDS[@]}")
    INDEX_OFFSET=261                          # labels start at 262; Set 7 ended at 261 (219-261 = 43 IDs)
  elif [ "$TASK_SET" = "all" ]; then
    TASK_IDS=("${DEFAULT_TASK_IDS[@]}" "${SECOND_RUN_TASK_IDS[@]}")
    INDEX_OFFSET=0                            # labels 01 … N across both sets
  else
    TASK_IDS=("${DEFAULT_TASK_IDS[@]}")
  fi
fi

# Auto-derive ECS "Started by" prefix from benchmark name if not explicitly set.
# -n/--limit runs use dispatch-hf{N} so ECS tasks show as dispatch-hf50-01 etc.
# swebench_pro_seal → "dispatch-seal"
# swebench_pro      → "dispatch-pro"
if [ -z "$STARTED_BY" ]; then
  if [ -n "$LIMIT" ]; then
    STARTED_BY="dispatch-hf${LIMIT}"
  else
    _bench="${BENCHMARK:-swebench_pro}"
    # strip common prefix so the label stays short (≤ 36 chars after "-NN")
    _short="${_bench#swebench_pro_}"   # "seal" for swebench_pro_seal
    [ "$_short" = "$_bench" ] && _short="${_bench#swebench_}"  # fallback strip
    STARTED_BY="dispatch-${_short}"
  fi
fi

TS=$(date -u +%Y%m%d-%H%M%S)
LOG=/tmp/smoke-pro-$TS
mkdir -p "$LOG"

# Auto-inject context env vars for the per-task Slack notification so the
# message rendering knows the SHA, task set, and who triggered the run.
# Cell name is injected per-cell inside the dispatch loop below since it
# changes per --cell flag. Explicit --extra-env always wins (we skip if
# the same key is already present from the caller).
if ! _has_extra "CROSSBENCH_IMAGE_SHA"; then
  EXTRA_ENV+=("CROSSBENCH_IMAGE_SHA=${SHA}")
fi
if ! _has_extra "CROSSBENCH_TASK_SET"; then
  EXTRA_ENV+=("CROSSBENCH_TASK_SET=${TASK_SET}")
fi
if ! _has_extra "CROSSBENCH_TRIGGERED_BY"; then
  _triggered_by="$(git config user.name 2>/dev/null || echo "${USER:-unknown}")"
  EXTRA_ENV+=("CROSSBENCH_TRIGGERED_BY=${_triggered_by}")
fi

echo "=== SWE-bench Pro smoke ==="
echo "  SHA            : $SHA"
echo "  Build ID       : ${BUILD_ID:-(none)}"
echo "  Started by     : ${STARTED_BY:-(unset)}"
echo "  Base profiles  : $BASE_PROFILES"
echo "  Cells          : ${#CELLS[@]}"
echo "  Reruns / cell  : $RERUNS"
echo "  Tasks / cell   : ${#TASK_IDS[@]}"
echo "  Total dispatches: $(( ${#CELLS[@]} * RERUNS ))"
echo "  S3 bucket      : $S3_BUCKET"
echo "  Timestamp      : $TS"
echo "  Logs           : $LOG/"
echo

# --- 1. Wait for build (optional) ---------------------------------------
if [ -n "$BUILD_ID" ] && [ "$SKIP_WAIT" -eq 0 ]; then
  echo "Waiting for CodeBuild $BUILD_ID …"
  while :; do
    STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" \
             --region us-east-1 \
             --query 'builds[0].buildStatus' --output text 2>/dev/null || echo "UNKNOWN")
    case "$STATUS" in
      SUCCEEDED) echo "  build SUCCEEDED"; break ;;
      FAILED|STOPPED|FAULT) echo "ERROR: build exited $STATUS"; exit 1 ;;
      *) echo "  status=$STATUS  (sleep 30)"; sleep 30 ;;
    esac
  done
  echo
fi

# --- 2. Build --task-id args once --------------------------------------
TID_ARGS=()
for t in "${TASK_IDS[@]}"; do TID_ARGS+=(--task-id "$t"); done

# --- 2b. Register the pinned task-def ONCE (avoid AWS rate limit) -------
# When the script fans out N parallel dispatches against the same SHA,
# every dispatch_swebench.py would otherwise call RegisterTaskDefinition
# concurrently and AWS rate-limits with "Too many concurrent attempts to
# create a new revision of the specified family" — losers get a
# botocore.errorfactory.ClientException, never launch any ECS tasks, and
# their parent prefix stays empty. Solution: register once here, pass the
# revision string to every subprocess via --task-def (which short-circuits
# the per-subprocess register step). Same pattern as dispatch_swebench.py.
echo "Registering pinned task-def once for $SHA…"
TASK_DEF=$(python3 -c "
import sys, boto3
short_sha = '$SHA'[:7]
session = boto3.Session(region_name='us-east-1')
ecs = session.client('ecs')
ecr = session.client('ecr')
try:
    ecr.describe_images(repositoryName='cross-bench-orchestrator',
                        imageIds=[{'imageTag': short_sha}])
except ecr.exceptions.ImageNotFoundException:
    sys.exit(f'ERROR: ECR tag :{short_sha} not found — build first.')
td = ecs.describe_task_definition(taskDefinition='cross-bench-dev-orchestrator')['taskDefinition']
pinned_image = f'564175165906.dkr.ecr.us-east-1.amazonaws.com/cross-bench-orchestrator:{short_sha}'
for c in td.get('containerDefinitions', []):
    if c.get('name') == 'orchestrator':
        c['image'] = pinned_image
        break
for k in ('taskDefinitionArn', 'revision', 'status', 'requiresAttributes',
           'compatibilities', 'registeredAt', 'registeredBy'):
    td.pop(k, None)
resp = ecs.register_task_definition(**td)
print(f'cross-bench-dev-orchestrator:{resp[\"taskDefinition\"][\"revision\"]}')
") || exit 1
echo "  Pinned task-def: $TASK_DEF"
echo

# --- 3. Fan out cells × reruns -----------------------------------------
declare -a RUN_IDS=()
declare -A CELL_OF_RUN=()  # run_id -> cell_name
declare -A REPS_OF_CELL=() # cell_name -> "rid1 rid2 rid3"

# Dispatch strategy:
#   - Different CELLS run in parallel (each cell is an independent experiment).
#   - RERUNS of the same cell run sequentially to avoid saturating the ECS
#     cluster. Each rerun dispatches 20 tasks; launching them concurrently
#     fills all ~60 cluster slots at once, leaving later reruns with
#     RESOURCE:MEMORY failures on the first shard.
PIDS=()
for CELL_SPEC in "${CELLS[@]}"; do
  NAME="${CELL_SPEC%%:*}"
  TAIL="${CELL_SPEC#*:}"
  PROFILES="${BASE_PROFILES},${TAIL}"
  # Per-cell ``startedBy`` so concurrent cells are visually
  # distinguishable in ECS. ``dispatch-hf1-baseline-01`` vs
  # ``dispatch-hf1-mem0v2-01`` — the cell suffix tells the
  # operator at a glance which experiment a task belongs to.
  # ECS caps ``startedBy`` at 63 chars; the longest realistic suffix
  # observed in repo is ``mem0v2_embedded_no_infer`` (~24 chars),
  # so the upper bound stays well under the cap. Cell names with
  # special characters that ECS rejects (only ``[a-zA-Z0-9_-]`` is
  # allowed) are passed through unmodified — by convention cell names
  # don't contain forbidden chars.
  CELL_STARTED_BY="$STARTED_BY"
  if [ -n "$STARTED_BY" ]; then
    # Cell identity: prefer an ``abl_*`` profile in the cell tail
    # (canonical ablation name), else the operator-supplied label.
    CELL_PROFILE_NAME=""
    IFS=',' read -ra TAIL_PROFS_ARR <<< "$TAIL"
    for _p in "${TAIL_PROFS_ARR[@]}"; do
      case "$_p" in
        abl_*) CELL_PROFILE_NAME="$_p"; break ;;
      esac
    done
    CELL_PROFILE_NAME="${CELL_PROFILE_NAME:-$NAME}"
    # Suffix: ``set{N}`` for multi-task dispatches, ``task`` for single-task
    # (the dispatcher's per-shard offset then makes it ``task-<offset>``).
    if [ "${#TASK_IDS[@]}" -eq 1 ]; then
      _STARTED_BY_SUFFIX="task"
    else
      _STARTED_BY_SUFFIX="set${TASK_SET}"
    fi
    CELL_STARTED_BY="dispatch-${CELL_PROFILE_NAME}-${_STARTED_BY_SUFFIX}"
  fi
  (
    for ((R=1; R<=RERUNS; R++)); do
      # Use hf{N} prefix when pulling from HuggingFace so runs are
      # distinguishable from curated smoke set runs in S3 and Slack.
      if [ -n "$LIMIT" ]; then
        RID="pro-hf${LIMIT}-${NAME}-r${R}-${TS}"
      else
        RID="pro-smoke-${NAME}-r${R}-${TS}"
      fi
      echo "  dispatching cell=$NAME rerun=$R run_id=$RID profiles=$PROFILES started_by=$CELL_STARTED_BY"
      STEP_ARGS=()
      [ -n "$MAX_AGENT_STEPS" ] && STEP_ARGS+=(--max-agent-steps "$MAX_AGENT_STEPS")
      BENCH_ARGS=()
      [ -n "$BENCHMARK" ] && BENCH_ARGS+=(--benchmark "$BENCHMARK")
      ENV_ARGS=()
      for kv in "${EXTRA_ENV[@]}"; do ENV_ARGS+=(--extra-env "$kv"); done
      # Per-cell context: the StartedBy uses CELL_PROFILE_NAME (preferring
      # an ``abl_*`` profile from TAIL); the Slack message uses the same
      # value so its "Cell:" bullet matches the StartedBy header.
      ENV_ARGS+=(--extra-env "CROSSBENCH_CELL_NAME=${CELL_PROFILE_NAME:-$NAME}")
      # -n/--limit uses HuggingFace first-N; --offset slices a window
      # (ds[OFFSET:OFFSET+LIMIT]). Otherwise use the curated task list.
      TASK_ARGS=("${TID_ARGS[@]}")
      if [ -n "$LIMIT" ]; then
        TASK_ARGS=(--limit "$LIMIT")
        [ -n "$OFFSET" ] && TASK_ARGS+=(--offset "$OFFSET")
      fi
      # ``--timeout 54000`` (15 h) overrides dispatch_swebench.py's 4 h
      # default. SEAL Pro per-task wallclock is ~5-15 min typical with
      # tail outliers approaching 60 min; at ``--tasks-per-shard 10``
      # one shard can chew through 10+ hours of sequential work on
      # tail-heavy task distributions. Bumped from the 4 h default to
      # 15 h for the layers ablation to fit ``tps=10`` comfortably.
      python scripts/dispatch_swebench.py \
        --sha "$SHA" --variant pro \
        --task-def "$TASK_DEF" \
        --profiles "$PROFILES" \
        "${TASK_ARGS[@]}" --tasks-per-shard "$TASKS_PER_SHARD" \
        --timeout 54000 \
        --run-id "$RID" "${STEP_ARGS[@]}" "${BENCH_ARGS[@]}" \
        "${ENV_ARGS[@]}" \
        ${CELL_STARTED_BY:+--started-by "$CELL_STARTED_BY"} \
        ${INDEX_OFFSET:+--index-offset "$INDEX_OFFSET"} > "$LOG/$RID.log" 2>&1
      echo "  done cell=$NAME rerun=$R"
    done
  ) &
  PIDS+=($!)
  # Also track RIDs for the aggregator (must happen in parent shell).
  # **Load-bearing**: this loop MUST mirror the RID construction inside
  # the dispatch worker subshell above. When ``-n / --limit`` is set,
  # the dispatcher uses ``pro-hf{LIMIT}-...`` instead of ``pro-smoke-...``.
  # Before this fix the tracker hardcoded ``pro-smoke-`` and the aggregator
  # then tried to fetch ``s3://.../pro-smoke-baseline-r1-<TS>/summary.json``
  # while the real path was ``s3://.../pro-hf1-baseline-r1-<TS>/summary.json``
  # — every fetch returned None, the CSV ended up with zero rows, and the
  # Slack notification rendered all ``—`` placeholders. Keep the two RID
  # constructions in sync; if you change one, change the other.
  for ((R=1; R<=RERUNS; R++)); do
    if [ -n "$LIMIT" ]; then
      RID="pro-hf${LIMIT}-${NAME}-r${R}-${TS}"
    else
      RID="pro-smoke-${NAME}-r${R}-${TS}"
    fi
    RUN_IDS+=("$RID")
    CELL_OF_RUN["$RID"]="$NAME"
    REPS_OF_CELL["$NAME"]="${REPS_OF_CELL[$NAME]:-} $RID"
  done
done

echo
echo "All ${#PIDS[@]} cell worker(s) launched (reruns within each cell are sequential)."
echo "Waiting for all cells to complete…"
for P in "${PIDS[@]}"; do
  wait "$P" || echo "  cell worker PID $P returned nonzero (check $LOG/)"
done
echo

# --- 4. Aggregate per cell × pull every aggregate_metric ---------------
# Build two CSVs:
#   <LOG>/runs_raw.csv      one row per (cell, rerun) with every metric
#   <LOG>/cells_summary.csv one row per cell with mean/std per metric
# Both also upload to S3 under runs/swebench_pro/_smoke_<TS>/ so the
# operator can navigate to a single place and pull both files.
echo "=== Aggregating into CSV ==="

cat > "$LOG/_aggregate.py" <<'PYEOF'
"""Aggregate per-cell / per-rerun summary.json files into 2 CSVs."""
import csv, json, os, statistics, subprocess, sys
from collections import OrderedDict

bucket = os.environ["S3_BUCKET"]
ts = os.environ["TS"]
log_dir = os.environ["LOG"]
prefix = os.environ.get("S3_PREFIX", "runs/swebench_pro/")
if not prefix.endswith("/"):
    prefix += "/"

# Parse "cell:rid1,rid2,rid3;cell2:rid4,..." from CELL_RIDS env var
cells = OrderedDict()
for chunk in os.environ.get("CELL_RIDS", "").split(";"):
    if not chunk:
        continue
    name, rids_csv = chunk.split(":", 1)
    cells[name] = [r for r in rids_csv.split(",") if r]

def fetch_summary(rid):
    try:
        r = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{prefix}{rid}/summary.json", "-"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except Exception as e:
        print(f"  [warn] fetch failed for {rid}: {e}", file=sys.stderr)
        return None

# Gather every (cell, rid) row with full aggregate_metrics
all_metric_keys = set()
raw_rows = []
for cell, rids in cells.items():
    for rid in rids:
        summ = fetch_summary(rid)
        if not summ:
            print(f"  [skip] no summary for {rid}", file=sys.stderr)
            continue
        metrics = summ.get("aggregate_metrics", {})
        all_metric_keys.update(k for k, v in metrics.items() if isinstance(v, (int, float)))
        raw_rows.append({
            "cell": cell,
            "run_id": rid,
            "status": summ.get("status", ""),
            "benchmark": summ.get("benchmark", ""),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })

metric_cols = sorted(all_metric_keys)
header_raw = ["cell", "run_id", "status", "benchmark"] + metric_cols

raw_path = f"{log_dir}/runs_raw.csv"
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header_raw, extrasaction="ignore")
    w.writeheader()
    for row in raw_rows:
        w.writerow(row)
print(f"Wrote {raw_path} ({len(raw_rows)} rows × {len(header_raw)} cols)")

# Per-cell summary: mean / std / n for each metric
summary_rows = []
for cell in cells:
    cell_rows = [r for r in raw_rows if r["cell"] == cell]
    row = {"cell": cell, "n_reruns": len(cell_rows)}
    for m in metric_cols:
        vals = [r[m] for r in cell_rows if r.get(m) is not None]
        if vals:
            row[f"{m}_mean"] = statistics.mean(vals)
            row[f"{m}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        else:
            row[f"{m}_mean"] = None
            row[f"{m}_std"] = None
    summary_rows.append(row)

summary_cols = ["cell", "n_reruns"]
for m in metric_cols:
    summary_cols.extend([f"{m}_mean", f"{m}_std"])

summary_path = f"{log_dir}/cells_summary.csv"
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)
print(f"Wrote {summary_path} ({len(summary_rows)} rows × {len(summary_cols)} cols)")

# Upload both to S3 under a navigable path.
s3_root = f"s3://{bucket}/{prefix}_smoke_{ts}/"
for local in (raw_path, summary_path):
    subprocess.run(["aws", "s3", "cp", local, s3_root + os.path.basename(local)],
                   capture_output=True)
print(f"Uploaded to: {s3_root}")

# Quick stdout preview — pass_rate mean ± std per cell.
print("\n=== Quick preview (pass_rate) ===")
for row in summary_rows:
    m = row.get("pass_rate_mean")
    s = row.get("pass_rate_std")
    n = row["n_reruns"]
    if m is None:
        print(f"  {row['cell']}: NO DATA (n={n})")
    else:
        print(f"  {row['cell']:<40} mean={m*100:6.1f}%  std={s*100:5.1f}pp  n={n}")
PYEOF

# Build CELL_RIDS env: "cell1:rid1,rid2;cell2:rid3,rid4;..."
CELL_RIDS=""
for CELL_SPEC in "${CELLS[@]}"; do
  NAME="${CELL_SPEC%%:*}"
  RIDS_STR=$(echo "${REPS_OF_CELL[$NAME]}" | tr ' ' ',' | sed 's/^,//;s/,$//')
  CELL_RIDS="${CELL_RIDS}${NAME}:${RIDS_STR};"
done

# Derive the S3 prefix from --benchmark (or fall back to swebench_pro)
RUNS_PREFIX="runs/${BENCHMARK:-swebench_pro}/"
S3_BUCKET="$S3_BUCKET" TS="$TS" LOG="$LOG" CELL_RIDS="$CELL_RIDS" \
  S3_PREFIX="$RUNS_PREFIX" python3 "$LOG/_aggregate.py"

echo
echo "All artifacts:  s3://$S3_BUCKET/${RUNS_PREFIX}"
echo "CSV summary:    s3://$S3_BUCKET/${RUNS_PREFIX}_smoke_$TS/cells_summary.csv"
echo "CSV raw:        s3://$S3_BUCKET/${RUNS_PREFIX}_smoke_$TS/runs_raw.csv"
echo "Local copies:   $LOG/cells_summary.csv  $LOG/runs_raw.csv"

# --- 5. Post results table to Slack (no-op if SLACK_WEBHOOK_URL unset) ------
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  echo
  echo "=== Posting results to Slack ==="
  python3 "$(dirname "$0")/post_smoke_to_slack.py" \
    --csv "$LOG/cells_summary.csv" \
    --sha "$SHA" \
    --benchmark "${BENCHMARK:-swebench_pro}" \
    --s3-root "s3://$S3_BUCKET/${RUNS_PREFIX}" \
    ${STARTED_BY:+--started-by "$STARTED_BY"}
else
  echo
  echo "(SLACK_WEBHOOK_URL not set — skipping Slack notification)"
fi
