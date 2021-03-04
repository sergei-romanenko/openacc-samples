---
marp: true
title: Mysteries of parallelization
size: 4:3

theme: gaia
color: #000

style: |
  section {
    background-color: #fff;
  }

  section.lead h1 {
    text-align: center;
  }
  section.lead h2 {
    text-align: center;
  }
  section.lead h3 {
    text-align: center;
  }
  section.lead h4 {
    text-align: center;
  }
  section.lead p {
    text-align: center;
  }

  section p, li {
    font-size1: 22pt;
    font-size: 0.92em;
  }

  img[alt=width50] {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
  }
  img[alt=c10em] {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 10em;
  }
  img[alt=c15em] {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 15em;
  }
  img[alt=c20em] {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 20em;
  }

---

<!-- paginate: false -->
<!--- _class: lead --->

# Распараллеливание вычислений

</br>

### С.А.Романенко</br>

ИПМ им. М.В. Келдыша РАН, Москва  
28 февраля 2021

---

<!-- paginate: true -->

### Застой в информатике

В информатике наблюдался застой где-то с середины
70-х годов.

Причины:

* Безраздельно доминировал один тип вычислительных устройств: фон-неймановская 
  машина (ФНМ).

* Необычайная гибкость фон-неймановской машины делала жизнь
программистов слишком лёгкой. 

---

### Специфические свойства ФНМ

![c10em](single-cpu.svg)

* Последовательное исполнение команд.
* Большая и однородная память.
* Данные неподвижно лежат в том месте, куда их положили.

---

### Специфические свойства ФНМ

* Последовательное исполнение команд.
  * Нет проблем с синхронизацией. Если надо решить 2 задачи, то
    решаем сначала 1-ю, а потом -- 2-ю (или наоборот).

* Большая и однородная память.
  * Из любой части программы можно в любой момент "протянуть руку" к любому 
    месту памяти.
  * Данные неподвижно лежат в том месте, куда их положили. Поэтому,
    одни данные могут ссылаться на другие. А с помощью ссылок можно
    создавать сложные структуры данных (списки, деревья).


---

### Параллельные вычислители

* Многопроцессорность (multiprocessing).
  * Много процессоров (на плате или в шкафу).
  * Память у процессоров, может быть, общая, а, может быть,
    и нет.
* Многоядерность (multi-core processsor).
  * Несколько процессоров (ядер) в одном чипе.
  * Память для всех ядер - общая.
* Векторность (SIMD = single instruction, multiple data).
  * Одна команда может исполняться над несколькими данными.

---

### Многопроцессорность (multiprocessing)

![c20em](multi-cpu.svg)

* Много процессоров (на плате или в шкафу).
* Память у процессоров, может быть, общая, а, может быть,
  и нет.

---

### Векторность (SIMD)

SIMD = single instruction, multiple data).

![c15em](simd.svg)

* Одна команда может исполняться над несколькими данными.

---

### А как же программируемая логика (ПЛИС, FPGA)?

А дело в том, что ПЛ, это **не архитектура**, а **средство
реализации** различных архитектур.

* Реализовать архитектуру можно только если она уже придумана.
* Если взять одну из известных архитектур, то для неё уже есть
  готовые реализации.
* Но можно придумать новую, неслыханную архитектуру - и реализовать
  прототип с помощью ПЛ. А потом - заказать чипы.

---

### Недостатки ПЛ с точки зрения программиста

* Изобретение новых архитектур - это не программирование,
а несколько иная специальность.
* А если использовать известную архитектуру, то реализовывать
её через ПЛ - нецелесообразно (дорого и медленно работает).
* Компиляция программ для известных архитектур происходит
за секунды, а синтез схем для ПЛ может тянуться часами.
(Нечеловеческие условия работы... :grimacing:)

---

### Почему жить стало интереснее?

* Изменилась "система ценностей" в теории сложности вычислений.
  * Раньше цель состояла в уменьшении числа операции (что
    автоматически уменьшало время выполнения). А теперь,
    время выполнения и число операций (= затраты электроэнергии) -
    не одно и то же.
* Методы и алгоритмы, которые являются "хорошими" при
  последовательных вычислениях, не обязательно являются таковыми
  в случае параллельных вычислений.

---

### Что можно "распараллеливать"?

* Программу.
  * Вставляем директивы для компилятора.
  * Реорганизуем программу.

* Алгоритм.
  * Выявляем скрытый параллелизм.
  * Реорганизуем алгоритм.
  * **Заменяем на другой алгоритм.**

* Метод.
  * Изменяем или заменяем метод.