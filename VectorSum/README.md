[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Лабораторная+работа+№2)](https://git.io/typing-svg)
<h1 align="center"> Сложение элементов вектора </h1>
 
Задача: реализовать алгоритм сложения элементов вектора <br>
Язык: C++ или Python <br>
Входные данные: Вектор размером 100..1000000 <br>
Выходные данные: сумма элементов вектора + время вычисления <br>

<h2> Реализация</h2>
CPU-реализация использует numba.njit для оптимизации функции суммирования.<br>
<br>
GPU-реализация:
Используется cuda.jit для вычислений на GPU.
Shared memory используется для частичных сумм внутри блока.
Итоговая сумма вычисляется путем суммирования результатов всех блоков.<br>

<h2> Результаты </h2>

# Результаты с малой размерностью

![image](https://github.com/user-attachments/assets/563a9c51-2755-4e05-a97c-c86c64533b8d)
<br>

|   Experiment |   CPU Time (s) |    Sum |
|-------------:|---------------:|-------:|
|            1 |     0.699282   | 499760 |
|            2 |     0.00140405 | 499718 |
|            3 |     0.00139761 | 499697 |
|            4 |     0.00140262 | 499641 |
|            5 |     0.00137353 | 499963 |


|   Experiment |   GPU Time (s) |    Sum |
|-------------:|---------------:|-------:|
|            1 |    0.24862     | 499760 |
|            2 |    0.000699282 | 499718 |
|            3 |    0.000706434 | 499697 |
|            4 |    0.000762939 | 499641 |
|            5 |    0.000659704 | 499963 |

# Результаты с увеличенной размерностью

![image](https://github.com/user-attachments/assets/32dc72ee-7ad7-484c-a7a5-f75b000fa36b)


|   Experiment |   CPU Time (s) |         Sum |
|-------------:|---------------:|------------:|
|            1 |      0.119553  | 4.99975e+06 |
|            2 |      0.0146492 | 5.00077e+06 |
|            3 |      0.0166838 | 5.00003e+06 |
|            4 |      0.015604  | 4.99976e+06 |
|            5 |      0.0158205 | 5.00094e+06 |


|   Experiment |   GPU Time (s) |         Sum |
|-------------:|---------------:|------------:|
|            1 |     0.221226   | 4.99975e+06 |
|            2 |     0.00407791 | 5.00077e+06 |
|            3 |     0.00413013 | 5.00003e+06 |
|            4 |     0.00411367 | 4.99976e+06 |
|            5 |     0.00408483 | 5.00094e+06 |

    
