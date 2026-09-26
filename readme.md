<p align="center">
  <img src="assets/emblem.png" alt="ML DAFE" width="160">
</p>

<h1 align="center">Машинное обучение</h1>

<p align="center">
  Годовой курс для студентов ФАЛТ МФТИ · осенний семестр 2026/27
</p>

<p align="center">
  <a href="https://t.me/mldafemipt">
    <img src="https://img.shields.io/badge/Telegram-@mldafemipt-0072CE?logo=telegram&logoColor=white" alt="Telegram">
  </a>
</p>

---

Первый семестр — классический ML, основы глубокого обучения и устройство ML-проекта.

## С чего начать

1. Прочитать [расписание](docs/schedule.md): темы, даты, что смотреть дома
2. Прочитать [правила оценивания](docs/grading.md): 40 процентов домашки, 30 лабораторные, 30 устный ответ, все по десятибалльной шкале
3. Поставить окружение по [инструкции](docs/setup.md)
4. Прочитать, [как проверяются лабораторные](docs/labs.md)
5. Открыть папку [первого занятия](classes/week_01_intro/)

Отдельно: [что изучать дальше](docs/resources.md) — восемьдесят проверенных
ссылок на курсы, книги и инструменты, от теории вероятностей до MLOps.

## Как устроен курс

Лекции вы смотрите дома, в аудитории мы работаем руками. Каждая суббота —
две пары. На первой разбираем тему вместе по семинарскому ноутбуку.
На второй вы решаете задачи сами.

Готовые лабораторные присылаете, они прогоняются через автоматическую
проверку. Вместе с оценкой приходит отчет: какие тесты прошли, какие упали
и почему.

За семестр три большие домашние работы. В первой вы работаете с настоящими
диалогами пользователей и языковых моделей: учите модели определять
реакцию пользователя на ответы, придумываете свою разбивку диалогов
на темы, честно валидируете модели и сохраняете так, чтобы ими мог
пользоваться кто-то другой. Во второй пишете генератор стихов по первой строке. Тема третьей
объявляется при выдаче.

Задачи взяты не из учебника. Это то, чем занимаются в индустрии люди,
которые готовят данные для больших моделей и проверяют их качество.

## Структура репозитория

```
docs/       расписание, оценивание, установка, что изучать дальше
classes/    материалы занятий: семинар и лабораторная на каждую неделю
homeworks/  три домашние работы
tools/      проверка окружения
assets/     эмблема курса
```

В папке каждого занятия четыре ноутбука: семинарский и лабораторный,
каждый в двух версиях. Версия без суффикса — та, с которой работаем
на паре. Версия `_solved` появляется после занятия.

## Программа первого семестра

| Часть | Занятия | О чем |
|:--|:--|:--|
| I. Классическое машинное обучение | 1–6 | kNN, наивный Байес, линейные модели, SVM, метрики и валидация, PCA, кластеризация, деревья и бустинг |
| II. Глубокое обучение | 7–10 | нейросети для изображений, эмбеддинги слов, рекуррентные сети, внимание, трансформер, ViT |
| III. ML-проект | 11–14 | как устроен проект: структура, шаблоны, конфиги, воспроизводимость |

## Чего ожидать

К декабрю вы должны уметь не «запускать модели», а отвечать на вопросы,
которые задают на собеседовании и на работе:

- почему эта метрика, а не другая
- откуда взялась эта тестовая выборка и можно ли ей верить
- не протекли ли данные из обучения в тест
- эта модель действительно лучше, или разница в пределах шума
- что сломается, когда данных станет в десять раз больше

Модели в этом списке нет ни в одном пункте. Это не случайно.

## Требования

Python от 3.11 до 3.13, менеджер окружений miniconda. Знание питона
на уровне «умею написать функцию и цикл». Теория вероятностей и линейная
алгебра в объеме программы.

Опыт в машинном обучении не нужен, курс начинается с нуля.

## Дополнительные материалы

Лекции, которые стоит посмотреть, и курсы, которые стоит пройти,
если хочется разбираться всерьез.

**К занятиям.** Основная серия — лекции [Лектория ФПМИ](https://www.youtube.com/playlist?list=PL4_hYwCyhAvZyW6qS58x4uElZgAkMVUvj)
по машинному обучению. Что смотреть к какому занятию — в [lectures.md](docs/lectures.md).

**Математика.**
[Теория вероятностей, Райгородский, МФТИ](https://www.youtube.com/playlist?list=PLthfp5exSWEqYroMZVPIOPd5Dz3ARAXzN) ·
[Основы статистики, Карпов](https://stepik.org/course/76) ·
[Математика для анализа данных](https://education.yandex.ru/handbook/math) ·
[Байесовские методы, Ветров](https://www.youtube.com/playlist?list=PLEqoHzpnmTfCiJpMPccTWXD9DB4ERQkyw)

**Машинное обучение.**
[Учебник ШАД](https://education.yandex.ru/handbook/ml) ·
[Курс Соколова, ВШЭ](https://github.com/esokolov/ml-course-hse) ·
[Stanford CS229](https://cs229.stanford.edu/) ·
[Statistical Rethinking](https://www.youtube.com/playlist?list=PLDcUM9US4XdNOlqSyhe38US8mFgmqzI14)

**Нейросети и зрение.**
[Deep Learning School, МФТИ](https://dls.samcs.ru) ·
[Deep learning на пальцах](https://dlcourse.ai/) ·
[Stanford CS231n](https://cs231n.stanford.edu/) ·
[Zero to Hero, Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

**Язык, звук, мультимодальность.**
[NLP Course For You, Войта](https://lena-voita.github.io/nlp_course.html) ·
[Speech course, ШАД](https://github.com/yandexdataschool/speech_course) ·
[Трек Speech, DLS МФТИ](https://www.youtube.com/playlist?list=PL0Ks75aof3TgnQ_q1AmIClOeX6c-F_Kw0) ·
[CMU 11-777, Multimodal](https://www.youtube.com/playlist?list=PLdGKJ8Bg_69xRZspmMOMmHDAf1nendPtA)

**Инженерия и MLOps.**
[MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) ·
[Made With ML](https://madewithml.com/courses/mlops/) ·
[CS329S, ML Systems](https://stanford-cs329s.github.io/)

**Инструменты.**
[PyTorch](https://pytorch.org/) ·
[Hugging Face](https://huggingface.co/) ·
[scikit-learn](https://scikit-learn.org/) ·
[Weights & Biases](https://wandb.ai/site) ·
[arXiv](https://arxiv.org/list/cs.LG/recent) ·
[Papers with Code](https://paperswithcode.com/)

**Запустить модель у себя.**
[Ollama](https://ollama.com/) ·
[LM Studio](https://lmstudio.ai/) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp)

Полный список с описаниями — в [resources.md](docs/resources.md),
восемьдесят проверенных ссылок.

## На чем основан курс

- [girafe-ai/ml-course](https://github.com/girafe-ai/ml-course), Радослав Нейчев и Владислав Гончаренко
- [ml-dafe/ml_mipt_dafe](https://github.com/ml-dafe/ml_mipt_dafe), курс машинного обучения на ФАЛТ
- [Учебник по машинному обучению от Яндекса](https://education.yandex.ru/handbook/ml)

Ноутбуки и задачи написаны для этого потока.

## Контакты

Чат курса в телеграме: **[@mldafemipt](https://t.me/mldafemipt)**.
Там объявления, ссылки на материалы и вопросы по задачам.

Туда же присылаются решенные лабораторные. Вопросы, которые не решаются
в чате, — лично на занятии. Проблемы с дедлайнами — заранее, а не после.
