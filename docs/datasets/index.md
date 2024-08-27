
# :material-database: Datasets

SleepKit provides support for a number of datasets to facilitate training the __sleep-monitoring tasks__. Most of the datasets are readily available and can be downloaded and used for training and evaluation. The datasets inherit from [Dataset](/sleepkit/api/sleepkit/datasets/dataset) and can be accessed either directly or through the factory singleton [`sk.DatasetFactory`](#dataset-factory).


## <span class="sk-h2-span">Available Datasets</span>

Below is a list of the currently available datasets in SleepKit. Please make sure to review each dataset's license for terms and limitations.

* **[MESA](./mesa.md)**: A longitudinal investigation of factors associated with the development of subclinical cardiovascular disease and the progression of subclinical to clinical cardiovascular disease in 6,814 black, white, Hispanic, and Chinese

* **[CMIDSS](./cmidss.md)**: The Child Mind Institute - Detect Sleep States (CMIDSS) dataset comprises 300 subjects with over 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep.

* **[YSYW](./ysyw.md)**: A total of 1,983 PSG recordings were provided by the Massachusetts General Hospital’s (MGH) Sleep Lab in the Sleep Division together with the Computational Clinical Neurophysiology Laboratory, and the Clinical Data Ani- mation Center.

* **[STAGES](./stages.md)**: The Stanford Technology Analytics and Genomics in Sleep (STAGES) study is a prospective cross-sectional, multi-site study involving 20 data collection sites from six centers including Stanford University, Bogan Sleep Consulting, Geisinger Health, Mayo Clinic, MedSleep, and St. Luke's Hospital.

* **[Bring-Your-Own-Data](./byod.md)**: Add new datasets to SleepKit by providing your own data. Subclass `Dataset` and register it with the `sk.DatasetFactory`.

---

## <span class="sk-h2-span">Dataset Factory</span>

The dataset factory, `sk.DatasetFactory`, provides a convenient way to access the datasets. The factory is a thread-safe singleton class that provides a single point of access to the datasets via the datasets' slug names. The benefit of using the factory is it allows registering new additional datasets that can then be leveraged by existing and new tasks.

The dataset factory provides the following methods:

* **sk.DatasetFactory.register**: Register a custom dataset
* **sk.DatasetFactory.unregister**: Unregister a custom dataset
* **sk.DatasetFactory.has**: Check if a dataset is registered
* **sk.DatasetFactory.get**: Get a dataset
* **sk.DatasetFactory.list**: List all available datasets
