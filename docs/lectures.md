# Лекции: что смотреть и когда

Лекции в аудитории не читаются. Занятие целиком уходит на разбор темы
в ноутбуке, а теория смотрится дома. Такой формат работает только при одном
условии: вы приходите, посмотрев лекцию. Иначе семинар превращается
в пересказ.

Все ссылки проверены на работоспособность 25 сентября 2026 года.

## Основная серия

Курс опирается на лекции **Лектория ФПМИ** по машинному обучению
(основной поток, 3 курс), [плейлист целиком](https://www.youtube.com/playlist?list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj).

| Лекция | Тема | К занятию | Ссылка |
|:--:|:--|:--:|:--|
| 1 | Introduction. Naive Bayes, kNN | 1, повтор | [8s9073kNXgY](https://www.youtube.com/watch?v=8s9073kNXgY&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 2 | Linear Regression | 2 | [GKIkGc2bnmU](https://www.youtube.com/watch?v=GKIkGc2bnmU&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 3 | Linear Classification & Logistic Regression | 3 | [_SUJsDLtJsE](https://www.youtube.com/watch?v=_SUJsDLtJsE&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 4 | SVM. PCA | 5 | [C9nop0dgJc4](https://www.youtube.com/watch?v=C9nop0dgJc4&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 5 | Cross validation. BVD | 4 | [-taRv4WTUbo](https://www.youtube.com/watch?v=-taRv4WTUbo&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 6 | Decision trees and Ensembles | 8 | [-0Do2_WIJAQ](https://www.youtube.com/watch?v=-0Do2_WIJAQ&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 7 | Gradient boosting | 9 | [hTECDpL_JYM](https://www.youtube.com/watch?v=hTECDpL_JYM&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 8 | Intro to Deep Learning | 10 | [wgWfAMhw5ik](https://www.youtube.com/watch?v=wgWfAMhw5ik&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 9 | Optimization and regularization in Deep Learning | 11–12, дополнительно | [XWAvufvzk2M](https://www.youtube.com/watch?v=XWAvufvzk2M&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 10 | Recurrent Neural Networks and Language Models | по желанию | [TkFBozjO72Y](https://www.youtube.com/watch?v=TkFBozjO72Y&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |
| 11 | Convolutional neural networks | 11; повтор к 13 | [6_vFNBwfK28](https://www.youtube.com/watch?v=6_vFNBwfK28&list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj) |

Занятия 10–13 — введение в нейронные сети. Лекция 10 про рекуррентные сети
оставлена для тех, кому интересно посмотреть самостоятельно.
К занятию 11 подготовлена [теория сверток, pooling, оптимизации и аугментаций](../classes/week_11_convolutions/lecture_11.ipynb).
Переход с NumPy на PyTorch и эксперименты на CIFAR-10 разбираются
в [семинаре 11](../classes/week_11_convolutions/seminar_11.ipynb).
В лекции рассмотрены BatchNorm и границы областей AdaptiveAvgPool.
Подробные примеры pooling, его стоимость, расстояние между центрами полей восприятия,
LayerNorm и GroupNorm собраны в
[конспекте о слоях](../classes/week_11_convolutions/notes_11_layers.ipynb). В семинаре сравнивают resize входа,
усреднение входа и усреднение карт признаков на изображениях разных размеров.
MaxPool и AvgPool проверяют на настоящем изображении CIFAR-10 и увеличенном окне одного канала.
В [лабораторной 11](../classes/week_11_convolutions/lab_11.ipynb) реализуют NumPy-свёртку с тремя и двумя циклами,
собственные MaxPool2d, Dropout и BatchNorm2d с режимами train/eval, затем CNN для CIFAR-10.
Девять обязательных задач дают 10 баллов; бонусы — свёртка с одним циклом и без циклов.
Каждый бонус даёт +1 балл при общем потолке оценки 11. Качество, скорость и устойчивость сравнивают в одном протоколе.
Матричная запись свертки, im2col и вывод градиентов собраны
в [дополнительном конспекте](../classes/week_11_convolutions/notes_11_convolution_matrix.ipynb).

К занятию 12 подготовлен [семинар по основам PyTorch](../classes/week_12_pytorch/seminar_12.ipynb).
Тензоры и сравнение с NumPy, перестановка осей картинки, перенос на CPU/CUDA/MPS,
Dataset и DataLoader, собственный Module и один шаг обучения разбираются маленькими ячейками.
Реальный граф autograd показан до и после backward, производные сверены вручную.
Отдельно рассмотрены train/eval, отключение градиентов, сохранение весов и tqdm.
В конце кратко сопоставлены PyTorch, TensorFlow и JAX; установка двух последних не требуется.

Занятие 14 про устройство ML-проекта идет без видео: к нему в репозитории
будет короткий текст, ссылка появится в расписании перед занятием.

## Дополнительная серия

**Машинное обучение 22f**, курс girafe-ai в более подробной версии.
Берите, если после основной лекции тема осталась непонятной, или наоборот,
если хочется глубже.

| Тема | Лекция | Семинар |
|:--|:--|:--|
| Повторение линейной алгебры | [vKfdtHnXVEY](https://youtu.be/vKfdtHnXVEY) | — |
| kNN | — | [Ha3pJJnt5YA](https://youtu.be/Ha3pJJnt5YA) |
| Линейная регрессия | [imzlM4jRbD4](https://youtu.be/imzlM4jRbD4) | [LLGLeM3JKDQ](https://youtu.be/LLGLeM3JKDQ) |
| Логистическая регрессия | — | [vSeETg1two8](https://youtu.be/vSeETg1two8) |
| SVM и PCA | [mlA-XxC9Ugg](https://youtu.be/mlA-XxC9Ugg) | [z-JqKoyHHRI](https://youtu.be/z-JqKoyHHRI) |
| Деревья и ансамбли | [kbNZsQj2eHk](https://youtu.be/kbNZsQj2eHk) | — |
| Градиентный бустинг | [Y97xrRiLY1Q](https://youtu.be/Y97xrRiLY1Q) | [4vo39B6M270](https://youtu.be/4vo39B6M270) |
| Введение в глубокое обучение | [G--msc2IR-Y](https://youtu.be/G--msc2IR-Y) | [0WMAfRuFHy8](https://youtu.be/0WMAfRuFHy8) |
| Обратное распространение ошибки | — | [HGk5xQ0azdo](https://youtu.be/HGk5xQ0azdo) |

Еще одна серия того же уровня — [«Тренировки по ML»](https://youtu.be/k3UJOG-DKHE)
Яндекса: тот же материал другими словами, полезно, если что-то не улеглось.

## Оптимизаторы: дополнительная лекция к занятиям 11–12

[«Машинное обучение 10. Нейросети, часть 2»](https://www.youtube.com/watch?v=oMrUaNQC3po)
Радослава Нейчева, 16 апреля 2019 года: SGD и momentum — примерно с 04:00,
Нестеров — с 07:50, AdaGrad — с 13:50, RMSprop — с 16:00, Adam — с 18:30.
Времена ориентировочные. Последующие разделы видео посвящены нормализации,
регуляризации и аугментациям.

В [лекции 11](../classes/week_11_convolutions/lecture_11.ipynb), раздел 8,
оставлен один виджет с 3D-траекториями девяти оптимизаторов на четырёх поверхностях.
Методы можно включать отдельно, поверхности - вращать мышью.
Формулы, механизмы обновлений и сравнение памяти вынесены в конспект. Подробный разбор с формулами,
кодом и графиками находится в [конспекте оптимизаторов](../classes/week_11_convolutions/notes_11_optimizers.ipynb):
AdamW, Adafactor, Lion, Muon/MuonClip, Shampoo/SOAP и экономия памяти состояний.
Видеолекция 2019 года не является источником утверждений о распространенности
методов в 2026 году: для современного обзора в ноутбуке указаны отдельные первичные источники.

В конспекте дополнительное сравнение десяти конфигураций CNN на CIFAR-10 включается
переменной `W11_OPT_CIFAR=1` до запуска ядра. Каждый метод получает три кандидата
шага на одном разбиении; выбор выполняется по validation, итоговый отчет — на test.
По умолчанию этот тяжелый опыт выключен, математические демонстрации выполняются всегда.
