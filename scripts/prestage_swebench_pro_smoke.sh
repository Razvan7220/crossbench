#!/usr/bin/env bash
#
# Pre-pull the 10 SWE-bench Pro smoke images from Docker Hub.
#
# These are the stratified smoke set (4 python / 3 go / 2 js / 1 ts,
# seed=42 sampling — see workflow.md "Pro smoke set"). Used to validate
# the Pro adapter wiring + the no_memory / mem0v2 schema spread
# before scaling to the full ablation grid.
#
# Cost: free (Docker Hub pulls). Disk: ~30-50 GB total (per image
# ~3-5 GB compressed). Time: ~10-15 min depending on bandwidth.
#
# Why direct Docker Hub instead of ECR mirror: the Pro mirror infra
# isn't deployed yet (Pulumi not applied on a maintainer's machine —
# see workflow.md "TODO — Pro mirror not yet deployed"). 10 anonymous
# pulls are well under Docker Hub's 100/6h cap so no mirror needed for
# the smoke. The full 731-instance grid will need the mirror.
#
# Usage:
#   bash scripts/prestage_swebench_pro_smoke.sh

set -euo pipefail

# Reproducible list — regenerate with:
#   python3 -c 'from datasets import load_dataset; ...' (see the
#   Pro-smoke-set table in workflow.md for the seed=42 sampling code).
IMAGES=(
  "jefzda/sweap-images:internetarchive.openlibrary-internetarchive__openlibrary-f343c08f89c772f7ba6c0246f384b9e6c3dc0add-v08d8e8889ec945ab821fb156c04c7"
  "jefzda/sweap-images:ansible.ansible-ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86"
  "jefzda/sweap-images:qutebrowser.qutebrowser-qutebrowser__qutebrowser-cc360cd4a34a126274c7b51f3b63afbaf3e05a02-v5fc38aaf22415ab0b70567368332beee7955b"
  "jefzda/sweap-images:internetarchive.openlibrary-internetarchive__openlibrary-25858f9f0c165df25742acf8309ce909773f0cdd-v13642507b4fc1f8d234172bf81299"
  "jefzda/sweap-images:flipt-io.flipt-flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a"
  "jefzda/sweap-images:gravitational.teleport-gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc"
  "jefzda/sweap-images:navidrome.navidrome-navidrome__navidrome-1e96b858a91c640fe64e84c5e5ad8cc0954ea38d"
  "jefzda/sweap-images:protonmail.webclients-protonmail__webclients-c6f65d205c401350a226bb005f42fac1754b0b5b"
  "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-51d8f3b195bddb13a13ddc0de110722774d9bb1b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "jefzda/sweap-images:tutao.tutanota-tutao__tutanota-f373ac3808deefce8183dad8d16729839cc330c1-v2939aa9f4356f0dc9f523ee5ce19d09e08ab979b"
)

echo "Pre-staging ${#IMAGES[@]} SWE-bench Pro smoke images from Docker Hub…"
echo "Disk needed: ~30-50 GB total"
echo

for i in "${!IMAGES[@]}"; do
  IMG="${IMAGES[$i]}"
  echo "[$((i+1))/${#IMAGES[@]}] $IMG"
  if docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "  already local — skipping"
    continue
  fi
  docker pull "$IMG"
done

echo
echo "Done. ${#IMAGES[@]} images staged locally."
echo "Run a smoke against one:"
echo "  cross-bench run -b swebench_pro -p haiku45 -p no_memory \\"
echo "    --task-id instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a \\"
echo "    --db-url none --artifacts-dir ./artifacts/pro-smoke"
