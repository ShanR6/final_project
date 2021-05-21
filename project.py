import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2, radius, mass1, mass2, K):
    """Аргументы функции:
    x1,y1,vx1,vy1 - координаты и компоненты скорости 1-ой частицы
    x2,y2,vx2,vy2 - ... 2-ой частицы
    radius,mass1,mass2 - радиус частиц и их массы (массы разные можно задавать,
    радиус для простоты взят одинаковый)
    K - коэффициент восстановления (K=1 для абсолютного упругого удара, K=0
    для абсолютно неупругого удара, 0<K<1 для реального удара)
    Функция возвращает компоненты скоростей частиц, рассчитанные по формулам для
    реального удара, если стокновение произошло. Если удара нет, то возвращаются
    те же значения скоростей, что и заданные в качестве аргументов.
    """
    r12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # расчет расстояния между центрами частиц
    v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)  # расчет модулей скоростей частиц
    v2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

    # проверка условия на столкновение: расстояние должно быть меньше 2-х радиусов
    if r12 <= 2 * radius:

        '''вычисление углов движения частиц theta1(2), т.е. углов между
        направлением скорости частицы и положительным направлением оси X.
        Если частица  покоится, то угол считается равным нулю. Т.к. функция
        arccos имеет область значений от 0 до Pi, то в случае отрицательных
        y-компонент скорости для вычисления угла theta1(2) надо из 2*Pi
        вычесть значение arccos(vx/v)
        '''
        if v1 != 0:
            theta1 = np.arccos(vx1 / v1)
        else:
            theta1 = 0
        if v2 != 0:
            theta2 = np.arccos(vx2 / v2)
        else:
            theta2 = 0
        if vy1 < 0:
            theta1 = - theta1 + 2 * np.pi
        if vy2 < 0:
            theta2 = - theta2 + 2 * np.pi

        # вычисление угла соприкосновения.
        if (y1 - y2) < 0:
            phi = - np.arccos((x1 - x2) / r12) + 2 * np.pi
        else:
            phi = np.arccos((x1 - x2) / r12)

        # Пересчет  x-компоненты скорости первой частицы
        VX1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
              * np.cos(phi) / (mass1 + mass2) \
              + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
              * np.cos(phi) / (mass1 + mass2) \
              + K * v1 * np.sin(theta1 - phi) * np.cos(phi + np.pi / 2)

        # Пересчет y-компоненты скорости первой частицы
        VY1 = v1 * np.cos(theta1 - phi) * (mass1 - K * mass2) \
              * np.sin(phi) / (mass1 + mass2) \
              + ((1 + K) * mass2 * v2 * np.cos(theta2 - phi)) \
              * np.sin(phi) / (mass1 + mass2) \
              + K * v1 * np.sin(theta1 - phi) * np.sin(phi + np.pi / 2)

        # Пересчет x-компоненты скорости второй частицы
        VX2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
              * np.cos(phi) / (mass1 + mass2) \
              + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
              * np.cos(phi) / (mass1 + mass2) \
              + K * v2 * np.sin(theta2 - phi) * np.cos(phi + np.pi / 2)

        # Пересчет y-компоненты скорости второй частицы
        VY2 = v2 * np.cos(theta2 - phi) * (mass2 - K * mass1) \
              * np.sin(phi) / (mass1 + mass2) \
              + ((1 + K) * mass1 * v1 * np.cos(theta1 - phi)) \
              * np.sin(phi) / (mass1 + mass2) \
              + K * v2 * np.sin(theta2 - phi) * np.sin(phi + np.pi / 2)

    else:
        # если условие столкновнеия не выполнено, то скорости частиц не пересчитываются
        VX1, VY1, VX2, VY2 = vx1, vy1, vx2, vy2

    return VX1, VY1, VX2, VY2


def collision_in_box(x1, y1, vx1, vy1, Lx, Ly, radius, K1):
    if (x1 <= (- Lx + radius) or x1 >= (Lx - radius)):
        VX = - K1 * vx1
    else:
        VX = vx1
    if (y1 <= (- Ly + radius) or y1 >= (Ly - radius)):
        VY = - K1 * vy1
    else:
        VY = vy1
    return VX, VY


def collision_with_wall(x1, y1, vx1, vy1, Ly, Lx1, Lx2, orientation, radius, K1):
    """Аргументы функции:
    x1, y1, vx1, vy1 - координаты и компоненты скорости 1-ой частицы
    Ly - x-я (для вертикальной) или y-я (для горизонтальной) координата
    стенки.
    Lx1, Lx2 - начальная и конечные y(x)-координаты стенки.
    orientation - ключ, задающий ориентацию стенки (0 - горизонтальная,
    1 - вертикальная)
    radius - радиус частицы
    K1 - коэффициент восстановления (K1=1 для абсолютного упругого удара, K1=0
    для абсолютно неупругого удара, 0<K1<1 для реального удара)
    Функция возвращает компоненты скорости частицы, рассчитанные по формулам для
    реального удара о стенку, если стокновение произошло. Если удара нет,
    то возвращаются те же значения скоростей, что и заданные в
    качестве аргументов.
    """
    if orientation == 0:
        r = y1 - Ly  # вычисление расстояния до стенки для горизонтальной ориентации
        dist = np.absolute(r)  # модуль расстояния
        key = 2
        if Lx1 <= x1 <= Lx2 and dist <= radius:  # проверка условия столкновения
            if vy1 < 0 and r <= 0:  # если частица отлетает от стенки вниз
                # считаем, что стокновения на самом деле нет
                key = 1
            if vy1 > 0 and r >= 0:  # если частица отлетает от стенки вверх, то
                # считаем, что стокновения на самом деле нет
                key = 1
            if vy1 < 0 and r > 0:  # если частица стремится пролететь сквозь стенку сверху вниз
                # то считаем, что это и есть стокновение
                key = 0
            if vy1 > 0 and r < 0:  # если частица стремится пролететь сквозь стенку снизу вверх
                # то считаем, что это и есть стокновение
                key = 0

        if Lx1 <= x1 <= Lx2 and key == 0:  # условие, при котором пересчитываются скорости
            VY = - K1 * vy1  # отражение от горизонтальной стенки
            VX = vx1  # х-я компонента скорости не меняется
        else:
            VY = vy1
            VX = vx1

    if orientation == 1:  # аналогичные операции для вертикальной ориентации стенки
        r = x1 - Ly
        dist = np.absolute(r)
        key = 2
        if Lx1 <= y1 <= Lx2 and dist <= radius:
            if vx1 < 0 and r <= 0:
                key = 1
            if vx1 > 0 and r >= 0:
                key = 1
            if vx1 < 0 and r > 0:
                key = 0
            if vx1 > 0 and r < 0:
                key = 0

        if Lx1 <= y1 <= Lx2 and key == 0:
            VX = - K1 * vx1
            VY = vy1
        else:
            VX = vx1
            VY = vy1

    return VX, VY


def circle_func(x_centre_point,  # х-координата центральной точки окружности
                y_centre_point,  # у-координата центральной точки окружности
                R):
    """ Функция, возвращающая точки окружности относительно определенного центра
    """
    x = np.zeros(30)  # Создание массива для координаты х
    y = np.zeros(30)  # Создание массива для координаты у
    for i in range(0, 30, 1):  # Цикл, определяющий множество точек окружности относительно центра
        alpha = np.linspace(0, 2 * np.pi, 30)
        x[i] = x_centre_point + R * np.cos(alpha[i])
        y[i] = y_centre_point + R * np.sin(alpha[i])

    return x, y


# ------------------- Задача -------------------------

# функция расчета движения шара
def move_func(s, t):
    x, y, v_x, v_y = s

    dxdt = v_x
    dv_xdt = 0

    dydt = v_y
    dv_ydt = - g

    return dxdt, dydt, dv_xdt, dv_ydt


# параметры шаров (берем одинаковые шарики)
radius = 0.5
mass = 0.5

# местонахождение вертикальных и горизонтальных стенок, которые располагаются
# симметрично
Lx = 10
Ly = 10

# ----------------------------------------------------------------------------------
# ТУТ ДОБАВЛЯЕМ РАЗЛИЧНЫЕ СТЕНКИ!!!!!!!!!!
# ----------------------------------------------------------------------------------

# Самая левая (верхняя) горизонтальная стенка
Ly01 = 0
Lx01 = -10
Lx02 = -8

# Первая, считая слева направо, вертикальная стенка
Ly11 = -8
Lx11 = 0
Lx12 = -2

# Вторая горизонтальная стенка
Ly21 = -2
Lx21 = -8
Lx22 = -6

# Вторая вертикальная стенка
Ly31 = -6
Lx31 = -2
Lx32 = -4

# Третья горизонтальная стенка
Ly41 = -4
Lx41 = -6
Lx42 = -4

# Третья вертикальная стенка
Ly51 = -4
Lx51 = -4
Lx52 = -6

# Четвёртая горизонтальная стенка
Ly61 = -6
Lx61 = -4
Lx62 = -2

# Четвёртая вертикальная стенка
Ly71 = -2
Lx71 = -8
Lx72 = -6

# Пятая горизонтальная стенка
Ly81 = -8
Lx81 = -2
Lx82 = 0

# Пятая вертикальная стенка
Ly91 = 0
Lx91 = -10
Lx92 = -8
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------


# коэффициенты упругости для парного столкновения шаров и столкновения со стенкой
K = 1
K1 = 1
g = 9.8

# время и число шагов
T = 70
n = 3000
tau = np.linspace(0, T, n)

# начальные условия для скоростей и координат шариков
x1, y1, vx1, vy1 = -9, 1, 0.5, 0
x2, y2, vx2, vy2 = -6, 9, 0, 0

# массивы для записи последующих координат
X1 = []
Y1 = []
X2 = []
Y2 = []

# разбиение интервала интегрирования на маленькие интервалы и интегрирование
for k in range(n - 1):
    t = [tau[k], tau[k + 1]]
    s1 = x1, y1, vx1, vy1
    sol1 = odeint(move_func, s1, t)
    x1 = sol1[1, 0]
    y1 = sol1[1, 1]
    vx1 = sol1[1, 2]
    vy1 = sol1[1, 3]

    s2 = x2, y2, vx2, vy2
    sol2 = odeint(move_func, s2, t)
    x2 = sol2[1, 0]
    y2 = sol2[1, 1]
    vx2 = sol2[1, 2]
    vy2 = sol2[1, 3]

    X1.append(x1)
    Y1.append(y1)
    X2.append(x2)
    Y2.append(y2)

    # проверка условий столкновения со стенками
    res1 = collision_in_box(x1, y1, vx1, vy1, Lx, Ly, radius, K1)
    res2 = collision_in_box(x2, y2, vx2, vy2, Lx, Ly, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    # ----------------------------------------------------------------------------------
    # ОБСЧИТЫВАЕМ СТОЛКНОВЕНИЕ С ДОБАВЛЕННЫЕМИ СТЕНКАМИ
    # ----------------------------------------------------------------------------------
    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly01, Lx01, Lx02, 0, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly01, Lx01, Lx02, 0, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly11, Lx12, Lx11, 1, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly11, Lx12, Lx11, 1, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly21, Lx21, Lx22, 0, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly21, Lx21, Lx22, 0, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly31, Lx32, Lx31, 1, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly31, Lx32, Lx31, 1, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly41, Lx41, Lx42, 0, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly41, Lx41, Lx42, 0, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly51, Lx52, Lx51, 1, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly51, Lx52, Lx51, 1, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly61, Lx61, Lx62, 0, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly61, Lx61, Lx62, 0, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly71, Lx72, Lx71, 1, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly71, Lx72, Lx71, 1, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly81, Lx81, Lx82, 0, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly81, Lx81, Lx82, 0, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]

    res1 = collision_with_wall(x1, y1, vx1, vy1, Ly91, Lx92, Lx91, 1, radius, K1)
    res2 = collision_with_wall(x2, y2, vx2, vy2, Ly91, Lx92, Lx91, 1, radius, K1)
    vx1, vy1 = res1[0], res1[1]
    vx2, vy2 = res2[0], res2[1]
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------

    # проверка условий столкновения шаров между собой
    res3 = collision(x1, y1, vx1, vy1, x2, y2, vx2, vy2, radius, mass, mass, K)
    vx1, vy1, vx2, vy2 = res3[0], res3[1], res3[2], res3[3]

# создание анимации
fig, ax = plt.subplots()

plt.xlim(-Lx, Lx)
plt.ylim(-Ly, Ly)

# отрисовка стенок
plt.plot([-Lx, -Ly], [Lx, -Ly], color='b')
plt.plot([Lx, -Ly], [Lx, Ly], color='b')
plt.plot([Lx, Ly], [-Lx, Ly], color='b')
plt.plot([-Lx, Ly], [-Lx, -Ly], color='b')

# ----------------------------------------------------------------------------------
# ТУТ РИСУЕМ ДОБАВЛЕННЫЕ СТЕНКИ
# ----------------------------------------------------------------------------------
plt.plot([Lx01, Lx02], [Ly01, Ly01], '-', color='b')
plt.plot([Ly11, Ly11], [Lx11, Lx12], '-', color='b')
plt.plot([Lx21, Lx22], [Ly21, Ly21], '-', color='b')
plt.plot([Ly31, Ly31], [Lx31, Lx32], '-', color='b')
plt.plot([Lx41, Lx42], [Ly41, Ly41], '-', color='b')
plt.plot([Ly51, Ly51], [Lx51, Lx52], '-', color='b')
plt.plot([Lx61, Lx62], [Ly61, Ly61], '-', color='b')
plt.plot([Ly71, Ly71], [Lx71, Lx72], '-', color='b')
plt.plot([Lx81, Lx82], [Ly81, Ly81], '-', color='b')
plt.plot([Ly91, Ly91], [Lx91, Lx92], '-', color='b')
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------

# инициализация шаров
ball1, = plt.plot([], [], 'o', color='r', ms=1)
ball2, = plt.plot([], [], 'o', color='r', ms=1)


# функция анимации
def animate(i):
    ball1.set_data(circle_func(X1[i], Y1[i], radius))
    ball2.set_data(circle_func(X2[i], Y2[i], radius))


ani = animation.FuncAnimation(fig, animate, frames=n - 1, interval=1)

plt.axis('equal')

ani.save('poluchilos1.gif')
