.venv:
	@python3 -m venv .venv

.venv/installed: requirements.txt | .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/installed

test: .venv/installed
	@.venv/bin/python -m pytest -v

run: .venv/installed
	@.venv/bin/python -m main data/posed_ims/posed_ims --out run1 --model dinov2_vitg14 --layer 31 --clusters 32

poses_to_cloud: .venv/installed
	@.venv/bin/python -m poses_to_cloud data/posed_ims/posed_ims test

build_cloud_colmap: .venv/installed
	@.venv/bin/python -m build_cloud_colmap data/posed_ims/posed_ims test --matcher sequential

view_cloud: .venv/installed
	@.venv/bin/python -m view_cloud test/cloud.ply

ssh:
	@ssh $(VM_USERNAME)@$(VM_IP)

.PHONY: test run ssh poses_to_cloud build_cloud_colmap view_cloud
