# PC Classification Improvement Report

## 1. Project Overview

This work improves the **PC (Plate Category) Classification** module of a Cambodian Automatic License Plate Recognition (ALPR) system.

The task contains **26 categories**, including Cambodian provinces and special plate types such as:

- Phnom Penh
- Kandal
- Siem Reap
- Kampot
- Battambang
- State
- Police
- RCAF
- Custom
- Other provincial categories

The main objective is to classify the correct plate category from a rectified license plate while preventing the model from relying on the plate number itself.

The final inference concept is:

```text
Vehicle image
    ↓
Plate detection
    ↓
Plate perspective correction
    ↓
PN detection
    ↓
Mask PN region
    ↓
PC classification
```

---

## 2. Main Problems in the Original PC Classification

The original classification workflow had several important limitations:

1. Severe class imbalance.
2. Many repeated photographs of the same physical plate.
3. Risk of train/validation/test leakage.
4. Synthetic images that did not fully match the real inference pipeline.
5. Synthetic images that were cleaner than difficult real images.
6. Very few real examples for many classes.
7. Risk that the model could memorize the plate number instead of province/category information.
8. Accuracy alone was misleading because Phnom Penh dominated the dataset.

The improvement therefore focused on the **whole data pipeline**, not only the neural network.

---

## 3. Master Dataset Audit

The rebuilding process started from the master annotated dataset.

```text
Images                : 1,525
PC annotations        : 1,525
PC classes            : 26
Valid PC polygons     : 1,525
Invalid PC polygons   : 0
Missing image files   : 0
```

The dataset was highly imbalanced. Examples:

```text
PC_PhnomPenh        : 1,046 images
PC_Custom           :   149 images
PC_SiemReap         :    62 images
PC_State            :    46 images
PC_Kandal           :    36 images
Several classes     :   1–3 images
```

This immediately showed that class-sensitive evaluation metrics would be required.

---

## 4. Plate Rectification

Each PC annotation contains a four-point polygon.

The polygon was perspective-corrected and normalized to:

```text
400 × 149 pixels
```

This transforms an angled plate into a standard rectangular classifier input.

```text
Angled plate
     ↓
Perspective transform
     ↓
400 × 149 normalized plate
```

### Concept

Perspective normalization removes unnecessary geometric variation and helps the classifier focus on useful information such as text, color, and layout.

---

## 5. Robust Quadrilateral Ordering

During preprocessing, strongly rotated plates exposed problems in the original corner-ordering logic.

The geometry code was improved and verified visually.

Final extraction result:

```text
Source images       : 1,525
Successfully created: 1,525
Failed              : 0
```

A key lesson was that automatic checks alone are not enough for computer vision preprocessing. Manual visual QA was also necessary.

---

## 6. Plate Number Masking

The plate number was treated as **nuisance information**.

Example:

```text
Original

┌─────────────────────────┐
│ កណ្តាល                  │
│        2B-1234          │
│ KANDAL                  │
└─────────────────────────┘

Masked

┌─────────────────────────┐
│ កណ្តាល                  │
│ █████████████████       │
│ KANDAL                  │
└─────────────────────────┘
```

The mask preserves:

- Khmer category text
- English category text
- plate background
- plate color
- layout information

while removing the plate number.

### Concept

The classifier should learn **category-related features**, not memorize plate numbers.

---

## 7. Real Data Visual QA

Contact sheets were generated to compare:

```text
RECTIFIED ORIGINAL | MASKED INPUT
```

The visual QA checked for:

- upside-down plates
- mirrored plates
- incorrect rectification
- severe stretching
- masks covering category text
- plate numbers remaining visible
- invalid mask geometry

This step caught preprocessing problems that simple numeric checks did not detect.

---

## 8. Physical Plate Group Analysis

Although the dataset contained:

```text
1,525 images
```

it represented only:

```text
847 unique physical plate groups
```

Therefore:

```text
678 images
```

were repeated observations.

Examples of repeated physical plates:

```text
PC_PhnomPenh__PN_2BU-4332   : 39 images
PC_PhnomPenh__PN_2BN-8837   : 29 images
PC_PhnomPenh__PN_2CN-9950   : 23 images
PC_Takeo__PN_2A-5615        : 16 images
```

### Concept

Multiple photos of the same physical plate are not independent training examples.

---

## 9. Leakage-Safe Grouped Split

A normal image-level random split could place photos of the same plate in different subsets.

```text
Wrong:

same plate photo 1 → TRAIN
same plate photo 2 → VALIDATION
same plate photo 3 → TEST
```

Instead, every physical plate group was assigned to only one split.

Final real split:

```text
TRAIN
670 groups
1,210 images
26 classes

VALIDATION
86 groups
175 images
18 classes

TEST
91 groups
140 images
23 classes
```

Leakage result:

```text
Group leakage = 0
```

### Concept

Group-aware splitting prevents artificially inflated model performance.

---

## 10. Extremely Rare Classes

Some classes contained only one or two independent real plate groups.

Examples:

```text
PC_KampongChhnang
PC_OddarMeanchey
PC_PreahVihear
PC_KohKong
PC_Kratie
PC_MondulKiri
PC_Ratanakiri
PC_TboungKhmum
```

Because a class with one real group cannot appear independently in train, validation, and test, the split prioritized preserving at least one real training example.

Final class coverage:

```text
TRAIN      : 26 / 26 classes
VALIDATION : 18 / 26 classes
TEST       : 23 / 26 classes
```

This limitation must be considered when interpreting per-class results.

---

## 11. Training Imbalance Analysis

After grouped splitting, the real training set remained highly imbalanced.

Examples:

```text
PC_PhnomPenh : 842
PC_Custom    : 126
PC_SiemReap  : 47
PC_State     : 39
PC_Kandal    : 23
Many classes : 1–2
```

Synthetic-data requirements were calculated only from the **training split**.

Candidate targets:

```text
Target 100/class → 2,158 synthetic images
Target 150/class → 3,382 synthetic images
Target 200/class → 4,632 synthetic images
Target 250/class → 5,882 synthetic images
```

A target floor of approximately **200 samples per class** was selected.

---

## 12. Measuring the Real PN Detector

The real PN detector:

```text
models/pn_locator/pn_bbx.pt
```

was evaluated on the **1,210 real training plates**.

Results:

```text
Detection rate   : 100.00%

IoU
P50              : 0.8872
P90              : 0.9338
P95              : 0.9442

Corner error
P50              : 7.55 px
P90              : 12.95 px
P95              : 15.46 px

Center X error
P50              : 0.55% plate width
P90              : 1.52%

Center Y error
P50              : 1.21% plate height
P90              : 3.02%

Prediction / GT area ratio
P50              : 1.0973
P90              : 1.2000
```

All predictions were:

```text
BBOX : 1,210
```

### Concept

Synthetic masks should imitate the actual detector behavior instead of using arbitrary mask shapes.

---

## 13. Real PN Geometry Analysis

Two dominant PN layouts were identified.

### Horizontal-like layout

```text
Center X ≈ 0.50
Center Y ≈ 0.58
Width    ≈ 0.96 plate width
Height   ≈ 0.56 plate height
```

Concept:

```text
┌──────────────────────────┐
│ Khmer text               │
│ ████████████████████████ │
│ English text             │
└──────────────────────────┘
```

### Side-like layout

```text
Center X ≈ 0.66
Center Y ≈ 0.50
Width    ≈ 0.63 plate width
Height   ≈ 0.75 plate height
```

Concept:

```text
┌──────────────────────────┐
│ Khmer   │ ██████████████ │
│ English │ ██████████████ │
└──────────────────────────┘
```

These measurements were used to improve synthetic generation.

---

## 14. Improved Synthetic Data Generation

Synthetic generation was changed from a fixed number per class to a deficit-based strategy:

```python
synthetic_needed = max(
    0,
    target_count - real_train_count
)
```

Two classes were excluded from synthetic generation in this experiment:

```text
PC_PhnomPenh
PC_Custom
```

Phnom Penh already had 842 real training images.

Custom was kept real-only because its plate layout required further study.

Final synthetic dataset:

```text
4,558 synthetic TRAIN images
```

---

## 15. Synthetic Generation Workflow

The improved generator follows the real inference process more closely:

```text
Generate plate background
        ↓
Render Khmer category text
        ↓
Render English category text
        ↓
Render random plate number
        ↓
Apply image degradation
        ↓
Simulate measured pn_bbx detector error
        ↓
Apply rectangular white PN mask
        ↓
Save classifier image
```

The random plate number is rendered before masking so imperfect detector masks can leave realistic residual number fragments.

---

## 16. Synthetic Image Degradation

Synthetic images were made less perfect through:

- Gaussian blur
- brightness variation
- contrast variation
- saturation variation
- sensor-like noise
- low-resolution simulation
- JPEG compression
- affine transformation
- perspective transformation

### Concept

This reduces the **synthetic-to-real domain gap**.

---

## 17. Synthetic Visual QA

Synthetic images were compared with real training images:

```text
REAL TRAIN | SYNTHETIC TRAIN
```

The QA checked:

- Khmer text
- English text
- layout
- mask geometry
- official plate colors
- side layout
- realism
- synthetic cleanliness compared with real images

The synthetic dataset was accepted after manual inspection.

---

## 18. Final Dataset

The final classification dataset contained:

```text
TRAIN = 5,768 images
    REAL      = 1,210
    SYNTHETIC = 4,558

VALIDATION = 175 images
    REAL ONLY

TEST = 140 images
    REAL ONLY
```

No synthetic image was included in validation or test.

---

## 19. Model Architecture

The classifier used:

```text
ImageNet-pretrained ResNet-18
```

ResNet is a type of CNN that uses residual connections.

```text
Basic CNN:
Conv → Conv → Pool → Dense

ResNet:
Input ──────────┐
  ↓             │
Conv → Conv     │
  ↓             │
  + ←───────────┘
  ↓
Next block
```

ResNet-18 was selected because:

- it is lightweight
- it supports transfer learning
- it is suitable for a relatively small dataset
- it trains efficiently on the RTX 4050 Laptop GPU
- it provides a strong baseline

Input size:

```text
416 × 160
```

This preserves the naturally wide shape of license plates better than a square input.

---

## 20. Fixed 26-Class Mapping

Because validation contains only 18 classes and test contains 23 classes, separate automatic class indexing could cause inconsistent labels.

One fixed 26-class mapping was therefore created from the training set and reused everywhere.

### Concept

Train, validation, test, and inference must use the same class-to-index mapping.

---

## 21. Class-Balanced Sampling

The final training set still contained:

```text
PC_PhnomPenh = 842
most augmented classes ≈ 200
```

Instead of deleting useful Phnom Penh data, a weighted sampler was used.

Each class received approximately equal total sampling probability.

### Concept

Balanced sampling reduces majority-class dominance without throwing away real data.

---

## 22. PC-v2 Baseline

The first ResNet-18 baseline used:

- ImageNet pretrained weights
- class-balanced sampling
- source-aware real/synthetic sampling
- augmentation
- label smoothing
- AdamW
- early stopping
- real validation Macro-F1 for model selection

Best validation result:

```text
Best REAL validation Macro-F1 = 0.7639
```

Baseline diagnostic test results:

```text
Accuracy          : 0.8571
Macro-F1          : 0.6101
Balanced Accuracy : 0.6401

Group Accuracy    : 0.9011
Group Macro-F1    : 0.6100
```

---

## 23. Why Macro-F1 and Balanced Accuracy Were Used

Raw accuracy was not enough because the dataset was highly imbalanced.

For example:

```text
PC_PhnomPenh = 83 / 140 test images
```

### Accuracy

Measures overall percentage of correct images.

### Macro-F1

Calculates F1 for each represented class and averages the classes equally.

### Balanced Accuracy

Calculates recall for each represented class and averages the recalls.

These metrics give rare classes more influence than standard accuracy alone.

---

## 24. Physical-Plate-Group Evaluation

Evaluation was also performed at physical-plate-group level.

Multiple images of the same plate were combined using majority voting.

### Concept

```text
Image-level evaluation
+
Physical-plate-level evaluation
```

gives a more informative result when repeated photographs exist.

---

## 25. Baseline Error Analysis

The baseline produced:

```text
Test images : 140
Correct     : 120
Errors      : 20
```

Error distribution:

```text
PC_Kandal        : 11
PC_KampongCham   : 1
PC_KampongSpeu   : 1
PC_KampongThom   : 1
PC_Kampot        : 1
PC_KohKong       : 1
PC_Kratie        : 1
PC_PreyVeng      : 1
PC_SiemReap      : 1
PC_Takeo         : 1
```

Therefore:

```text
55% of all test-image errors came from Kandal.
```

The most common confusion was:

```text
PC_Kandal → PC_PhnomPenh
```

---

## 26. Kandal Domain-Gap Analysis

The following were compared visually:

```text
REAL Kandal TRAIN
SYNTHETIC Kandal TRAIN
REAL Kandal VALIDATION
REAL Kandal TEST
```

Synthetic Kandal images were generally:

- clean
- sharp
- high contrast
- easy to read

The difficult real Kandal test plate was:

- blurry
- faded
- low contrast
- heavily masked
- difficult to read

This revealed a clear **synthetic-to-real domain gap**.

Another important observation was that the 12 Kandal test images came from only **one physical plate group**.

Therefore:

```text
11 wrong Kandal images
```

does not mean 11 independent Kandal plates failed.

---

## 27. PC-v2.1 Improvement Strategy

The ResNet-18 architecture was kept unchanged.

This allowed the experiment to measure the effect of better data sampling and augmentation.

Two major changes were introduced.

### 27.1 Dynamic Real/Synthetic Sampling

```text
15+ real images
→ 50% REAL
→ 50% synthetic

5–14 real images
→ 40% REAL
→ 60% synthetic

2–4 real images
→ 25% REAL
→ 75% synthetic

1 real image
→ 15% REAL
→ 85% synthetic

No synthetic data
→ 100% REAL
```

For Kandal:

```text
23 real
177 synthetic
```

PC-v2.1 sampled approximately:

```text
50% REAL
50% synthetic
```

rather than allowing synthetic examples to dominate.

### 27.2 Source-Aware Augmentation

Real images received mild augmentation.

Synthetic images received stronger:

- blur
- color variation
- perspective variation
- affine transformation
- contrast variation
- brightness variation

### Concept

Synthetic data should **support real data**, not replace its influence.

---

## 28. PC-v2.1 Training Result

Training used early stopping based on real validation Macro-F1.

Result:

```text
Best epoch             : 19
Best REAL Val Macro-F1 : 0.8003
```

Compared with baseline:

```text
PC-v2   : 0.7639
PC-v2.1 : 0.8003
```

Improvement:

```text
+0.0364
```

or approximately:

```text
+3.64 percentage points
```

---

## 29. PC-v2.1 Diagnostic Test Results

The improved model achieved:

```text
Accuracy                : 0.8857
Macro-F1                : 0.7602
Balanced Accuracy       : 0.7814

Physical-Group Accuracy : 0.9341
Group Macro-F1          : 0.7538
Group Balanced Accuracy : 0.7754
```

---

## 30. Final Improvement Comparison

| Metric | PC-v2 | PC-v2.1 | Improvement |
|---|---:|---:|---:|
| Best Real Validation Macro-F1 | 0.7639 | **0.8003** | **+0.0364** |
| Diagnostic Test Accuracy | 0.8571 | **0.8857** | **+0.0286** |
| Diagnostic Test Macro-F1 | 0.6101 | **0.7602** | **+0.1501** |
| Diagnostic Test Balanced Accuracy | 0.6401 | **0.7814** | **+0.1413** |
| Physical-Group Accuracy | 0.9011 | **0.9341** | **+0.0330** |
| Physical-Group Macro-F1 | 0.6100 | **0.7538** | **+0.1438** |

The strongest improvement was:

```text
Macro-F1

0.6101
   ↓
0.7602
```

This indicates that the improved workflow increased performance across represented classes, not only the majority class.

---

## 31. Remaining Limitation

Kandal remained difficult:

```text
Support   : 12 images
Precision : 1.0000
Recall    : 0.0833
F1        : 0.1538
```

Most Kandal errors were still:

```text
PC_Kandal → PC_PhnomPenh
```

However, all 12 images came from one difficult physical plate group.

More independent real Kandal examples are required before drawing a strong conclusion about Kandal performance.

---

## 32. Evaluation Limitation

The test-set errors were inspected while developing PC-v2.1.

Therefore, the current test set is better described as a **diagnostic evaluation set** rather than a completely untouched final benchmark.

For a final research-quality estimate, a future experiment should use:

- a fresh grouped holdout set, or
- grouped cross-validation, or
- newly collected real plates.

---

# 33. Complete Improved Workflow

```text
MASTER DATASET
      │
      ▼
1. Audit annotations
      │
      ▼
2. Extract PC polygon
      │
      ▼
3. Robust quadrilateral ordering
      │
      ▼
4. Perspective rectification
      │
      ▼
5. Normalize to 400 × 149
      │
      ▼
6. Transform PN annotation
      │
      ▼
7. Mask PN region
      │
      ▼
8. Visual QA of real data
      │
      ▼
9. Identify physical plate groups
      │
      ▼
10. Leakage-safe grouped split
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
    TRAIN           VAL            TEST
      │           REAL ONLY       REAL ONLY
      ▼
11. Analyze training imbalance
      │
      ▼
12. Measure pn_bbx detector errors
      │
      ▼
13. Analyze real PN geometry
      │
      ▼
14. Generate synthetic TRAIN data
      │
      ▼
15. Real-vs-synthetic visual QA
      │
      ▼
16. Build final dataset
      │
      ▼
17. ImageNet-pretrained ResNet-18
      │
      ▼
18. Class-balanced sampling
      │
      ▼
19. Source-aware real/synthetic sampling
      │
      ▼
20. Source-aware augmentation
      │
      ▼
21. Train classifier
      │
      ▼
22. Select checkpoint using
    REAL validation Macro-F1
      │
      ▼
23. Real-data evaluation
      │
      ▼
24. Physical-group evaluation
      │
      ▼
25. Visual error analysis
      │
      ▼
26. Improve domain alignment
      │
      ▼
PC-v2.1 BEST MODEL
```

---

# 34. Main Concepts Applied

The improved PC classification workflow applies the following concepts:

### Data-Centric AI
Improving data quality, preprocessing, balancing, and evaluation instead of only increasing model complexity.

### Perspective Normalization
Transforming plate polygons into a standard rectangular representation.

### Nuisance Feature Removal
Masking the plate number so the model learns category information.

### Data Leakage Prevention
Keeping all photographs of the same physical plate in one split.

### Group-Aware Splitting
Using physical plate identity instead of individual image identity when splitting.

### Class Imbalance Handling
Using synthetic generation and balanced sampling to reduce majority-class dominance.

### Synthetic Data Augmentation
Increasing training coverage for classes with very little real data.

### Domain Randomization
Adding blur, compression, noise, lighting, and geometric variation.

### Domain Gap Analysis
Comparing real and synthetic examples to identify unrealistic synthetic patterns.

### Transfer Learning
Using ImageNet-pretrained ResNet-18 instead of training a CNN completely from scratch.

### Weighted Sampling
Giving each class approximately equal training probability.

### Source-Aware Sampling
Giving real and synthetic examples different sampling probabilities.

### Source-Aware Augmentation
Applying stronger augmentation to synthetic images than real images.

### Macro-F1
Evaluating all represented classes more equally.

### Balanced Accuracy
Averaging recall across represented classes.

### Group-Level Evaluation
Evaluating independent physical plates in addition to individual images.

### Error Analysis
Inspecting actual mistakes before deciding how to improve the system.

### Early Stopping
Stopping training when real validation Macro-F1 stops improving.

---

# 35. Main Achievement

The main achievement was not simply changing the neural-network architecture.

The largest improvement came from rebuilding the entire PC classification pipeline with:

- validated annotations
- robust plate rectification
- PN masking
- automatic and manual visual QA
- physical-plate grouping
- leakage-safe splitting
- real-only validation and testing
- real detector-error measurement
- real PN geometry analysis
- deficit-based synthetic generation
- realistic synthetic augmentation
- transfer learning
- class-balanced sampling
- source-aware sampling
- source-aware augmentation
- Macro-F1-based model selection
- physical-group evaluation
- visual error analysis

The final PC-v2.1 model achieved:

```text
Best REAL Validation Macro-F1 : 0.8003
Diagnostic Test Accuracy      : 0.8857
Diagnostic Test Macro-F1      : 0.7602
Balanced Accuracy             : 0.7814
Physical-Group Accuracy       : 0.9341
Physical-Group Macro-F1       : 0.7538
```

Current best checkpoint:

```text
models/pc_v2/pc_resnet18_v21_best.pt
```

---

# 36. Conclusion

The PC classification improvement demonstrates that classification performance depends strongly on the quality of the entire machine-learning pipeline.

The main gains came from improving:

```text
data quality
+ geometric normalization
+ leakage prevention
+ class balance
+ synthetic realism
+ real/synthetic sampling
+ evaluation methodology
```

rather than simply increasing neural-network complexity.

The most important conclusion is:

> **Improving the data pipeline, training distribution, and evaluation strategy can produce larger gains than simply using a larger model.**

The current PC-v2.1 model is the recommended classifier for the next ALPR integration stage:

```text
Vehicle
   ↓
Plate detection
   ↓
Plate rectification
   ↓
PN detection and masking
   ↓
PC-v2.1 classification
   ↓
Province / plate-category prediction
```
