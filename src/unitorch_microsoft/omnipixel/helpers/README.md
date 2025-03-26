# Helpers


## Out-Painting

#### One Step
```bash
python3 -m unitorch_microsoft.omnipixel.scripts.stable_flux outpainting --data_file ./data.tsv --names "image;prompt" --prompt_col prompt --image_col image --processor_name p3 --cache_dir ./result1 --ratios 1.9 --do_opencv_inpainting True
```

#### Two Steps 1
```bash
python3 -m unitorch_microsoft.omnipixel.helpers.outpainting.step_expansion1 --data_file ./data.tsv --names "image;prompt" --prompt_col prompt --image_col image --cache_dir ./result1 --do_opencv_inpainting True
```