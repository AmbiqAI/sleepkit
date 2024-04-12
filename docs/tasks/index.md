# Tasks

## <span class="sk-h2-span">Introduction</span>

SleepKit provides several built-in __sleep-monitoring__ related tasks. Each task is designed to address a unique aspect such as sleep staging and sleep apnea detection. The tasks are designed to be modular and can be used independently or in combination to address specific use cases. In addition to the built-in tasks, custom tasks can be created by extending the `SKTask` base class and registering it with the task factory.

---

## <span class="sk-h2-span">Available Tasks</span>

### <span class="sk-h2-span"> [Detect](./detect.md)</span>

Sleep detection is the process of identifying sustained sleep/inactivity bouts. This task is useful for identifying long-term sleep patterns and for monitoring sleep quality.

### <span class="sk-h2-span">[Stage](./stage.md)</span>

Sleep stage classification is the process of identifying the different stages of sleep such as light, deep, and REM sleep. This task is useful for monitoring sleep quality and for identifying sleep disorders.

### <span class="sk-h2-span">[Apnea](./apnea.md)</span>

Sleep apnea detection is the process of identifying hypopnea/apnea events. This task is useful for identifying sleep disorders and for monitoring sleep quality.

<!-- ### <span class="sk-h2-span">[Arousal](./arousal.md)</span>

Sleep arousal detection is the process of identifying sleep arousal events. This task is useful for identifying sleep disorders and for monitoring sleep quality. -->

### <span class="sk-h2-span">[Bring-Your-Own-Task (BYOT)](./byot.md)</span>

Bring-Your-Own-Task (BYOT) is a feature that allows users to create custom tasks by extending the `SKTask` base class and registering it with the task factory. This feature is useful for addressing specific use cases that are not covered by the built-in tasks.

---


!!! Example "Recap"

    === "Detect"

        ### Sleep Detection

        Detect sustained sleep/inactivity bouts. <br>
        Refer to [Sleep Detect](./detect.md) for more details.

    === "Stage"

        ### Sleep Stage Classification

        Perform 2, 3, 4, or 5 stage sleep detection.
        Refer to [Sleep Stages](./stage.md) for more details.

    === "Apnea"

        ### Sleep Apnea Detection
        Detect hypopnea/apnea events. <br>
        Refer to [Sleep Apnea](./apnea.md) for more details.

    === "Arousal"

        ### Sleep Arousal Detection
        Detect sleep arousal events. <br>
        Refer to [Sleep Arousal](./arousal.md) for more details.

---
