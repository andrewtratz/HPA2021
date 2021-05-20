# HPA2021

Solo Kaggle HPA 2021 competition entry by Andrew Tratz. **Final score 0.530 (12th place)**.

Solution write-up as posted on Kaggle:

## Introduction
First of all, thank you to the organizers for hosting a fun (and very intense) competition. This is the second computer vision project I’ve attempted and I learned a tremendous amount along the way.

## Overall observations and approach
I began the project by reading twenty or thirty papers on the state-of-the-art for weakly-labeled image classification and segmentation, including the papers shared on the forum as well as quite a few others that I came across. I kept a daily journal of my progress and maintained a separate “to-do” list with ideas I had as I went along. Whenever I felt stuck I could always go back and find new directions to pursue.

When I entered the competition, @bestfitting had a commanding lead over the rest of the entrants. Since @bestfitting won the first HPA competition, I realized that he was probably re-using a substantial amount of his prior solution to achieve this result. His code and models were published and open sourced after the prior competition, so I resolved to use this as my starting point as well rather than forking any of the public kernels.

My first (working) baseline was to take one of @bestfitting’s pretrained models and simply re-map all of the category predictions from HPAv1 to HPAv2. I applied the same category to every cell in each image equally without attempting to localize. There was no category equivalent to “Negative” in the HPAv1 set so I copied these labels from one of the public kernels. Without doing any training whatsoever this baseline put me in front of all of the public kernels available at that time.

Despite the organizers’ attempts to pick high single-cell variation, the same-slide category correlation between cells remains extremely high. As such, I viewed the deciding factors in this competition as likely to be, in descending order:

1. Classification
2. Localization (i.e. differentiating cells on a single image)
3. Segmentation
Segmentation is last for two reasons: 1. The HPA cell segmentator is already quite robust and any significant deviation from its output is likely to be severely penalized and 2. The lack of any cell-level ground truth labels doomed, in my opinion, any attempt to use Mask-RCNN or other segmentation models to define cell boundaries. I’ll be surprised if any top teams had success with this.

Accordingly, I spent a significant amount of time in the early stages of the competition refining my image-level classifier, which I suspect was the opposite of what most teams were doing at that point. Once @h053473666’s kernel was published this changed the competition dynamic significantly as many other teams started integrating image-level classifiers into their ensembles.

## My top solution

### Input:

4-channel RGBY images from both train and external. I used a multi-threaded file downloader to get all of the external images early on (even so, this took several days of downloading). No attempt to remove any duplicates, etc.

My image classifier resized all images to 768x768 with no additional preprocessing.

My cell classifier used HPA cell segmentation first then padded each cell to square before masking and cropping each cell to 768x768.

### Train augmentation:

Fairly standard flipping horizontal, vertical, and transpose along with random cropping. This seemed more than adequate for my purposes.

### Segmentation

I used an implementation of HPA segmentator which allowed batch processing through the network, as the default implementation seems to process images one at a time. I modified the labeling function to run on 25% scaled masks as well to achieve significant time savings, albeit at a very slight hit to accuracy.

I implemented heuristics to exclude edge cells based on the size of the cell and nucleus relative to the average size of the cells and nuclei in the image. The discussion forums suggested this wasn’t very important but I found it to make a very significant difference in my score.

### Architecture:

I used @bestfitting’s architecture from the HPAv1 competition with only minor tweaks. This uses a DenseNet121 backbone followed by:

Average Pooling and Max Pooling concatenated together (Side note: the model appeared to rely mostly on the Max Pooling and used Average Pooling to regularize the output)
Batch Norm and 50% Dropout
A FC layer reducing from 2048 to 1024
ReLU, BN, Dropout 50%
A second FC layer reducing from 1024 -> 19
I also used one of @bestfitting’s pretrained models as a starting point for fine-tuning, which undoubtedly reduced my energy bill a bit. 😊

I used an identical architecture for both the image and cell classifiers and used the same pretrained model for both as well.

### Loss functions

I experimented with various combinations of loss functions. Whereas @bestfitting combined three loss functions into a single aggregate loss for training, I tried training models based on a single loss then ensembling them together.

### Training

Mid-way through the competition I switched to using mixed-precision training, which gave me a huge speed boost with very little coding required. I’ll be doing this for sure in all future competitions. I trained on a Quadro RTX 5000 and RTX 2080S. Running in parallel I could only use half of the Quadro RTX’s memory due to their mismatched memory sizes. A single epoch of my cell classifier takes 12-13 hours on this setup, and I wasn’t able to run as many epochs as I would have liked. (It took about 20 hours on full precision).

### Test augmentation

I used similar test-time augmentation to my train augmentation, including random cropping. This proved to be very useful, so I did as much as I could, limited by Kaggle’s 9 hour restriction. I wasn’t very diligent in setting random seeds to this introduced some variation from run to run. While the variation caused me some uncertainty in picking models it also made me less likely to cherry-pick models based on a single “lucky roll.”

### Heuristics

Right before the end of competition I realized that multiplying my image-level predictions against my cell-level predictions gave better output versus averaging them. The fact that all of the confidences shrink to near zero is no problem, since the relative ranking is the only important factor for the mAP metric. This allowed my model to more effectively down-weight many of the spurious single-cell predictions which didn’t align with the image-level prediction.

For the Negative category I used the max of the cell-level or image-level, which I think was better than the average (but I didn’t spend submissions to prove this for sure).

I also implemented a heuristic to penalize predictions from cells with a high Negative probability, applying a penalty to all of the non-Negative labels for this cell. This gave me a small score boost, largely because I believe my Negative label predictions are quite accurate.

### Ensemble

768x768 4-channel Cell classifier trained 4 epochs with BCE (6x random crop and augment) 50% weight
768x768 4-channel Cell classifier trained 4 epochs with BCE then 1 epoch with Focal Loss (6x random crop and augment) 50% weight

768x768 4-channel Image classifier trained 25 epochs with BCE (24x random crop and augment)

Negative category based on maximum of Cell or Image classifier. Other categories are Cell x Image result, downweighted by Negative class prediction.

### Results:

**Public LB: 0.547 (equivalent to 14th place on the leaderboard)
Private LB: 0.530 (12th place)**

My best public LB of 0.552 was based on a quick submit which I couldn’t reproduce on Kaggle. I think this 0.005 score difference from my next best submission was due to the reduced scaling of the segmentator labeling that I had to adopt for speed reasons (or perhaps a lucky combination of crops during my test augmentation).

## What worked:

* The model architecture was quite robust to overfitting and required very little additional hyperparameter tuning to get good results.
* Using @bestfitting’s pretrained weights saved me a lot of computation time
* FocalLoss generally gave superior results for image classification compared to other loss functions
* My heuristics (edge cell pruning, multiplicative ensembling, and penalizing Negative cells’ other class predictions) all gave me meaningful score boosts
* Mixed precision training was a life-saver
* Test augmentation – I never realized how important this was prior to this competition!
* Getting an early start on private LB submissions – I missed out on a silver medal in a previous competition because of submission-related problems and wanted to make sure that never happens again. I probably spent ten days or more focused on performance-tuning and bug-fixing my code to where it can consistently submit without problems. It also made me realize that going in with a crazy ensemble with everything under the sun wasn’t going to be a viable solution and I should focus more on ensembling together a few high-performing models.

## Other things I tried which (kind of) worked but I abandoned along the way:

* Focal Loss for cell-level classification seemed to quickly overfit. Not really sure why. This was particularly the case in early training. As a result, I pretrained my model using BinaryCrossEntropy to first reach a fairly stable point and then used Focal loss to fine tune further.
* Manual labelling of mitotic spindle cell instances was important to ensure quality labels for this class as most images only included 1-3 examples.
* L2 loss was surprisingly effective, particularly when ensembled with other losses. It didn’t make my final ensemble, however. (Actually, one of the models which I didn’t pick for the final submission used L2 and scored slightly higher, so I wish I’d kept this in).
* Increasing resolution to 1536x1536 for the image classifier. This showed some promise but the accuracy gains were minimal (probably some overfitting going on). It was also unacceptably slow during inference so I reverted back to 768x768 resolution.
* I built a model using single-channel green 224x224 nuclei crops focused just on nuclear categories (plus mitotic spindle). This worked but its predictions weren’t as strong as my final cell-level model’s. I think it might have struggled to differentiate nucleoplasm from other “diffuse field” categories, lacking broader context. Didn’t want to waste my precious remaining submissions to see if it was helping for any of the other categories.
* PuzzleCAM – I built a fully-functioning PuzzleCAM implementation on top of a modified version of the architecture I’m using. (Had to remove MaxPooling, etc. for this to function).

I used raw un-normalized feature outputs from the PuzzleCAM rather than true “CAMs” and used the feature overlaps with individual cell masks to make cell-level predictions.
I really liked this approach since the PuzzleCAM model is smart enough to make highly localized predictions while also being aware of the overall image context.
However after transitioning to a pure cell-level image classifier I was able to surpass my PuzzleCAM score significantly and abandoned this approach. I think with additional work this would have been an extremely powerful addition to my ensemble, but I just ran out of time.
The main downside of PuzzleCAM is it requires some additional heuristics to generate class predictions, so there’s a bit of fine-tuning involved.
I thought about building a custom loss function for PuzzleCAM which could take masks and images as input and generate losses at the cell level. This seemed like a novel approach to the problem but I didn’t get time to implement it.
* Rank-based ensembling – since the mAP metric is scale-invariant, the specific confidence scores output by a model are irrelevant. When combining loss functions with very different confidence ranges (e.g. like combining L2 together with Focal Loss) it sometimes seemed to help to ensemble together based on rank rather than confidence scores. However this caused slight performance degradation in my final ensemble so I deactivated it.
* EfficientNet backbone as a swap in replacement – I got the backbone working but probably didn’t have enough time remaining in the competition to train it sufficiently well.

## Other things I tried which didn’t work

* Applying weights to different categories generally seemed to destabilize my training
* ROCStar loss function (https://github.com/iridiumblue/roc-star) trying to optimize directly for AUC. Seemed to work for the first epoch but overall was less effective than other loss functions.
* I spent a lot of time building an artificial dataset by hand-curating cell examples from different categories and combining them using random copy-paste into completely artificial images. I thought this could substitute for the lack of ground truth labels since I could “create my own.” Ultimately, though, this was a recipe for rapid, massive overfitting.

It did, however, provide an interesting dataset for cross-validation of my image classifiers as I could estimate my mAP breakdown for each category.
However, trying to use this information to optimize my ensemble on a category level didn’t seem to give any meaningful improvement.
* I tried running PCA and an SVM to create a metric model using the cell images as well as their spectral domain counterparts, but my initial tests didn’t show promising results so I abandoned this approach.
* Multiprocessing in Kaggle kernels was extremely frustrating. Wasted lots of GPU quota trying to get this to work consistently. After one too many random crashes I quit doing this.

## Other things I wanted to do but didn’t have time

* I wanted to build a metric learning model looking at cell-level similarity, along the lines of what @bestfitting did for the HPAv1 competition but on the cell-level rather than image-level. Ultimately though I decided to spend my last few days of GPU processing time doing more cell-level epochs instead (probably a mistake?).
* Further refinement of the segmentation output – finding accidental blue stains, cells with multiple nuclei, mitotic spindles which are incorrectly segmented as two cells, etc.
* Mixup or copy/paste augmentation to increase frequency of rare class examples in the training dataset.
* CLAHE or other contrast-enhancement on the green channel.
* Teaming: I debated a lot about whether to team or not. However when the deadline approached I was struggling a lot with performance issues, multiprocessing instability, Kaggle errors, etc. and was stressed out just about getting my models working in the 9 hour time limit. There’s no doubt in my mind that if I had teamed up, even if just to ensemble my final output, my ranking would have improved. Lesson learned!

# How to utilize this code:

Please use the latest release for copies of the pretrained weights and CV files.

## Hardware requirements

I trained these models using a Quadro RTX 5000 and RTX 2080 Super on a single system. Other configurations should be feasible, although batch sizes may need to be modified. Training leverages mixed precision so older GPUs may require some minor code modification to work properly.

## Steps to build and test models

1. Set up environment according to requirements.txt
2. Copy all of the HPA competition data to your hard drive.
3. Download all of the HPA external images (see https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg for details) and copy to the same directory as the train images
4. Ensure the cross-validation metadata files (random_train_cell_cv0.csv, random_train_cv0.csv, random_valid_cell_cv0.csv, random_valid_cv0.csv) are in the project root directory
5. Segment all train and external images:

**python segment.py --metadata \<path to train.csv\> --root_dir \<path to train images\> --nuc_model \<path to nucleus model from HPACellSegmentator\> --cell_model \<path to cell model from HPACellSegmentator\> --meta_dir \<path to write new metadata\> --crop_dir \<path to write cropped train cell images\> --img_dir \<path to write resized input images\> --cache_dir \<path for a temporary cache directory for cell masks\>**

6. Segment all test images:

**python segment.py --metadata \<path to sample_submission.csv\> --root_dir \<path to test images\> --nuc_model \<path to nucleus model from HPACellSegmentator\> --cell_model \<path to cell model from HPACellSegmentator\> --meta_dir \<path to write new metadata\> --crop_dir \<path to write cropped test cell images\> --img_dir \<path to write resized test images\> --cache_dir \<path for a temporary cache directory for cell masks\>**

7. Train cell model for six epochs

**python train.py --out_dir Cell_BCE --loss BCE --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --scheduler AdamShortRun --epochs 6 --num_classes 19 --img_size 768 --crop_size 512 --batch_size 17 --effective_batch 34 --split_name random_folds10 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --cell cell --resume \<path to BFpretrain.pth\> --preload bestfitting --result_dir \<base directory where model outputs should be stored\> --img_path \<path to the cropped cell images\>

8. Train cell model for one epoch using Focal Loss

**python train.py --out_dir Cell_Focal --loss Focal --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --scheduler AdamShortRun --epochs 6 --num_classes 19 --img_size 768 --crop_size 512 --batch_size 17 --effective_batch 34 --split_name random_folds10 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --cell cell --resume \<result_dir\>\\models\\Cell_BCE\\fold0\\004.pth --preload self --result_dir \<result_dir\> --img_path \<path to the cropped cell images\>**

9. Train image model using Focal Loss

**python train.py --out_dir Image_Focal --loss Focal --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --scheduler AdamAndrew35 --epochs 35 --num_classes 19 --img_size 768 --crop_size 512 --batch_size 17 --effective_batch 34 --split_name random_folds10 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --cell image --resume \<path to BFpretrain.pth\> --preload bestfitting --result_dir \<result_dir\> --img_path \<path to the resized images\>**

10. Run inference using the three models

**python test.py --base_dir \<result_dir\> --out_dir Cell_BCE --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --predict_epoch 6 --augment all --num_classes 19 --img_size 768 --crop_size 512 --batch_size 30 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --seed 1 --dataset test --img_dir \<path to test cell crops\> --cellorimage cell**

**python test.py --base_dir \<result_dir\> --out_dir Cell_Focal --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --predict_epoch 6 --augment all --num_classes 19 --img_size 768 --crop_size 512 --batch_size 30 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --seed 1 --dataset test --img_dir \<path to test cell crops\> --cellorimage cell**

**python test.py --base_dir \<result_dir\> --out_dir Image_Focal --gpu_id \<IDs of GPUs to use\> --arch class_densenet121_dropout --predict_epoch \<best epoch\> --augment all --num_classes 19 --img_size 768 --crop_size 512 --batch_size 30 --fold 0 --workers 8 --in_channels 4 --puzzle 0 --seed 1 --dataset test --img_dir \<path to resized images\> --cellorimage image**

11. Reformat the prediction outputs

**python convertpredictions.py --dataset test --augment all --model Cell_BCE --epoch 6 --cellorimage cell --seeds 1 --rank 0 --predictions_path \<result_dir\>\\submissions --meta_csv_path \<path sample_submission.csv\> --cell_meta_csv_path \<path to metadata file generated during Segmentation step\> --out_path \<path to output location -- use project directory by default\> --add_datetime 0 --batch_size 100**

**python convertpredictions.py --dataset test --augment all --model Cell_Focal --epoch 6 --cellorimage cell --seeds 1 --rank 0 --predictions_path \<result_dir\>\\submissions --meta_csv_path \<path sample_submission.csv\> --cell_meta_csv_path \<path to metadata file generated during Segmentation step\> --out_path \<path to output location -- use project directory by default\> --add_datetime 0 --batch_size 100**

**python convertpredictions.py --dataset test --augment all --model ImageFocal --epoch \<best epoch\> --cellorimage image --seeds 1 --rank 0 --predictions_path \<result_dir\>\\submissions --meta_csv_path \<path sample_submission.csv\> --cell_meta_csv_path \<path to metadata file generated during Segmentation step\> --out_path \<path to output location -- use project directory by default\> --add_datetime 0 --batch_size 100**

12. Edit ensemble_kaggle016.csv to point to the paths of the 3 outputs of the prior step

13. Ensemble and generate submission.csv

**python ensemble.py \<path to ensemble_kaggle016.csv\> --dataset test --submission 1 --rank 0 --meta_csv \<path sample_submission.csv\> --cell_meta_csv_path \<path to metadata file generated during Segmentation step\> --out_path \<path to output location\> --add_datetime 0 --batch_size 100**

Thank you! Feel free to share any comments or feedback you have on this solution or its description.




