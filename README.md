Compare:

python -m cli.symph-compare \
  --a 'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif' \
  --b 'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_press_E_2018/32Nitrogen_Background.tif' \
  --out out/compare/porpoise_vs_nitrogen \
  --nodata "-9999"

Predict:

python -m cli.symph-predict \
  --target 'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif' \
  --predictors \
    'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_press_E_2018/32Nitrogen_Background.tif' \
    'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_press_E_2018/15Noise_125Hz_Shipping_20181122.tif' \
    'data/aligned/naturvarden-.och-belastningar-ostersjon-2018/National_press_E_2018/13Infrastructure.tif' \
  --out out/predict/porpoise_demo \
  --standardize --sample 200000 --alpha 1.0 --test_size 0.2 --nodata "-9999"


Check grid alignment:
python scripts/check_grid_alignment.py \
  --ref data/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif \
  "data/**/*.tif" \
  --out reports/grid_alignment_report.csv

Align or copy to directory:
python scripts/align_or_copy_to_dir.py \
  --ref data/naturvarden-.och-belastningar-ostersjon-2018/National_eco_E_2018/01Porpoise_Baltic.tif \
  --in-root data/raw \
  --out-root data/aligned_and_copied \
  --report reports/alignment_actions.csv \
  --overwrite