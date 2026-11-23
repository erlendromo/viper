.venv:
	@python3.14 -m venv .venv

.venv/installed: requirements.txt | .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/installed

test: .venv/installed
	@.venv/bin/python -m pytest -v

run: .venv/installed
	@.venv/bin/python -m main data/posed_ims/posed_ims data/IHM-dataset/mapped --out run4 --queries-json data/IHM-dataset/IHM_full_with_queries-fixed.json --id-mode index

poses_to_cloud: .venv/installed
	@.venv/bin/python -m poses_to_cloud data/posed_ims/posed_ims test

build_cloud_colmap: .venv/installed
	@.venv/bin/python -m build_cloud_colmap data/posed_ims/posed_ims test --matcher sequential

view_cloud: .venv/installed
	@.venv/bin/python -m view_cloud test/cloud.ply

ssh:
	@ssh $(VM_USERNAME)@$(VM_IP)



anyloc: .venv/installed
	@.venv/bin/python src/anyloc_full.py --db-imgs data/latest/imgs --db-poses data/latest/poses --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out run_anyloc 2>&1 | tee run_anyloc.log

salad: .venv/installed
	@.venv/bin/python src/baselines_full.py --model salad --db-imgs data/latest/imgs --db-poses data/latest/poses --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out run_salad 2>&1 | tee run_salad.log

megaloc: .venv/installed
	@.venv/bin/python src/baselines_full.py --model megaloc --db-imgs data/latest/imgs --db-poses data/latest/poses --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out run_megaloc 2>&1 | tee run_megaloc.log

compare_runs: .venv/installed
	@.venv/bin/python src/compare_runs.py run_megaloc run_megaloc_ft run_megaloc_ft_b2 run_megaloc_ft_info --out comparison_finetuning.txt

siglip: .venv/installed
	@.venv/bin/python src/siglip_text.py --db-imgs data/latest/imgs --db-poses data/latest/poses --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --megaloc-run run_megaloc_ft --out run_siglip 2>&1 | tee run_siglip.log



smoke_train: .venv/installed
	@.venv/bin/python src/train_pose.py --db-imgs data/latest/imgs --db-poses data/latest/poses --baseline-run run_megaloc --out run_ft_test --dry-run

train: .venv/installed
	@.venv/bin/python src/train_pose.py --db-imgs data/latest/imgs --db-poses data/latest/poses --baseline-run run_megaloc --out run_ft_info --pos-min-dist 0.3 --pos-min-angle 15 --pos-min-gap 200 --keyframe-dist 0.15 --keyframe-angle 10 2>&1 | tee run_ft_info.log

score: .venv/installed
	@.venv/bin/python src/baselines_full.py --model megaloc --weights run_ft_info/best.pth --db-imgs data/latest/imgs --db-poses data/latest/poses --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out run_megaloc_ft_info

.PHONY: test run ssh poses_to_cloud build_cloud_colmap view_cloud anyloc_full salad megaloc compare_runs siglip smoke_train train score
