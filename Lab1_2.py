import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import describe
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


def interval_grouping_abs_freq(n: int):
    interval_numbers = int(1 + math.log2(n)) #количество интервалов
    hist, bin_edges = np.histogram(sample, bins=interval_numbers) #интервальные границы
    sum_absolute_freq = np.sum(hist) #сумма абсолютных частот
    print(f"Сумма абсолютных частот: {sum_absolute_freq:.2f}")
    #построение диаграммы абсолютных частот
    plt.hist(sample, bins=bin_edges, edgecolor='k', alpha=0.7)
    plt.xlabel('Значения')
    plt.ylabel('Абсолютные частоты')
    plt.title('Диаграмма абсолютных частот')
    plt.show()
    return sum_absolute_freq, hist, bin_edges


def interval_grouping_rel_freq(n, hist, bin_edges):
    relative_freq = hist / n
    sum_relative_freq = np.sum(relative_freq)  #сумма относительных частот
    print(f"Сумма относительных частот: {sum_relative_freq:.2f}")
    #построение диаграммы относительных частот
    plt.bar(bin_edges[:-1], relative_freq, width=np.diff(bin_edges), edgecolor='k', alpha=0.7)
    plt.xlabel('Интервалы')
    plt.ylabel('Относительные частоты')
    plt.title('Диаграмма относительных частот')
    plt.show()
    return relative_freq


def visualization_1(relative_freq):
    plt.bar(bin_edges[:-1], relative_freq, width=np.diff(bin_edges), edgecolor='k', alpha=0.7, label='Гистограмма') #построение гистограммы относительных частот
    #построение теоретической кривой распределения
    x = np.linspace(min(sample), max(sample), 1000)
    pdf = stats.norm.pdf(x, loc=a, scale=np.sqrt(variance))
    plt.plot(x, pdf, 'r', label='Теоретическое распределение')
    plt.xlabel('Значения')
    plt.ylabel('Относительные частоты')
    plt.title('Гистограмма и теоретическое распределение')
    plt.legend()
    plt.show()


def visualization_2(sample, sum_absolute_freq):
    plt.hist(sample, bins=bin_edges, edgecolor='k', alpha=0.7, label='Гистограмма') #построение гистограммы абсолютных частот
    #построение графика теоретической частоты
    x = np.linspace(min(sample), max(sample), 1000)
    pdf = stats.norm.pdf(x, loc=a, scale=np.sqrt(variance))
    plt.plot(x, pdf * sum_absolute_freq * np.diff(bin_edges)[0], 'r', label='Теоретическое распределение')
    plt.xlabel('Значения')
    plt.ylabel('Абсолютные частоты')
    plt.title('Гистограмма абсолютных частот и теоретическое распределение')
    plt.legend()
    plt.show()
    return x


def visualization_3(sample):
    ecdf = ECDF(sample)
    # Построение графика эмпирической функции распределения
    x = np.linspace(min(sample), max(sample), 1000)
    plt.step(ecdf.x, ecdf.y, label='Эмпирическая функция распределения', color='b')
    # Построение графика теоретической функции распределения
    cdf = stats.norm.cdf(x, loc=a, scale=np.sqrt(variance))
    plt.plot(x, cdf, 'r', label='Теоретическая функция распределения')
    plt.xlabel('Значения')
    plt.ylabel('Вероятность')
    plt.title('Эмпирическая и теоретическая функции распределения')
    plt.legend()
    plt.show()


def visualization_4(sample):
    #построение бокс-плота
    plt.boxplot(sample, vert=False, widths=0.7)
    plt.xlabel('Значения')
    plt.title('Бокс-плот распределения')
    plt.show()
    #статистическая интерпретация
    mean_sample = np.mean(sample)
    std_dev_sample = np.std(sample)
    lower_limit = mean_sample - 3 * std_dev_sample
    upper_limit = mean_sample + 3 * std_dev_sample
    #подсчет числа выбросов в выборке
    outliers = [x for x in sample if x < lower_limit or x > upper_limit]
    expected_outliers = len(outliers)
    print(f"Ожидаемое число выбросов: {expected_outliers}")

def hit_in_k_interval(k, bin_edges, variance, a, relative_freq):
    interval_bounds = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)] #границы интервалов
    k_min, k_max = interval_bounds[k - 1]
    k_probability = norm.cdf(k_max, loc=a, scale=np.sqrt(variance)) - norm.cdf(k_min, loc=a, scale=np.sqrt(variance))
    print(f"Оценка вероятности попадания в {k}-ый интервал: { k_probability:.4f}")
    x = np.linspace(a - 3 * np.sqrt(variance), a + 3 * np.sqrt(variance), 1000)
    pdf = norm.pdf(x, loc=a, scale=np.sqrt(variance))
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label='Плотность вероятности')
    ax.axvline(k_min, color='red', linestyle='--')
    ax.axvline(k_max, color='red', linestyle='--')
    plt.show()
    fig2, ax = plt.subplots()
    ax.bar(bin_edges[:-1], relative_freq, width=np.diff(bin_edges), align='edge', alpha=0.7, label='Гистограмма относительных частот')
    ax.fill_between(bin_edges[:-1], relative_freq, where=[(k_min <= xi <= k_max) for xi in bin_edges[:-1]], alpha=0.3, color='red')
    plt.show()
    cdf_in_right_boundary = norm.cdf(k_max, loc=a, scale=np.sqrt(variance))
    exact_cdf_value = norm.cdf(k_max, loc=a, scale=np.sqrt(variance))
    print(
        f"Оценка значения интегральной функции распределения в правой границе {k}-ого интервала: {cdf_in_right_boundary:.4f}")
    print(f"Точное значение интегральной функции распределения в правой границе {k}-ого интервала: {exact_cdf_value:.4f}")


def laplas_probability(mean_x, variance, q):
    probability = norm.cdf(q + mean_x, loc=mean_x, scale=np.sqrt(variance)) - norm.cdf(-q + mean_x, loc=mean_x, scale=np.sqrt(variance))
    print(f"Вероятность: {probability:.4f}")


def relative_freq_events(sample, mean_x, q, n):
    count_within_interval = len([x for x in sample if -q + mean_x <= x <= q + mean_x])
    relative_freq_probability = count_within_interval / n
    print(f"Оценка вероятности по выборке: {relative_freq_probability:.4f}")


def sample_x50(a, n, variance):
    sample = np.random.normal(loc=a, scale=np.sqrt(variance), size=n)
    interval_numbers = int(1 + math.log2(n))  # количество интервалов
    hist, bin_edges = np.histogram(sample, bins=interval_numbers)  # интервальные границы
    sum_absolute_freq = np.sum(hist)  # сумма абсолютных частот
    visualization_2(sample, sum_absolute_freq)


def point_estimates_distribution_parameters_1(sample):
    mean_estimate = np.mean(sample)
    median_estimate = np.median(sample)
    variance_estimate = np.var(sample, ddof=1)  # ddof=1 для использования выборочной дисперсии
    std_deviation_estimate = np.std(sample, ddof=1)  # ddof=1 для использования выборочного стандартного отклонения
    skewness_estimate = (1 / std_deviation_estimate * 3) * np.mean((sample - mean_estimate) * 3)
    kurtosis_estimate = (1 / std_deviation_estimate * 4) * np.mean((sample - mean_estimate) * 4) - 3
    print(f"Математическое ожидание: {mean_estimate:.4f}")
    print(f"Медиана: {median_estimate:.4f}")
    print(f"Дисперсия: {variance_estimate:.4f}")
    print(f"Стандартное отклонение: {std_deviation_estimate:.4f}")
    print(f"Коэффициент ассиметрии: {skewness_estimate:.4f}")
    print(f"Эксцесс: {kurtosis_estimate:.4f}")


def point_estimates_distribution_parameters_2(sample):
    describe_results = describe(sample)
    print(f"Математическое ожидание: {describe_results.mean:.4f}")
    print(f"Медиана: {np.median(sample):.4f}")
    print(f"Дисперсия: {describe_results.variance:.4f}")
    print(f"Стандартное отклонение: {np.sqrt(describe_results.variance):.4f}")
    print(f"Коэффициент ассиметрии: {describe_results.skewness:.4f}")
    print(f"Эксцесс: {describe_results.kurtosis:.4f}")



n = 60
a = 0
variance = 9
k = 3
q = 1.75
sample = np.random.normal(loc=a, scale=np.sqrt(variance), size=n) # генерация выборки
sum_absolute_freq, hist, bin_edges = interval_grouping_abs_freq(n) # пункт 1.1
relative_freq = interval_grouping_rel_freq(n, hist, bin_edges) # пункт 1.2
visualization_1(relative_freq) # пункт 2.1
visualization_2(sample, sum_absolute_freq) # пункт 2.2
visualization_3(sample) # пункт 2.3
visualization_4(sample) # пункт 2.4
hit_in_k_interval(k, bin_edges, variance, a, relative_freq) #вопросы
laplas_probability(a, variance, q) #пункт 3.1
relative_freq_events(sample, a, q, n) #пункт 3.2
sample_x50(a, 50*n, variance) #пункт 3.3
print("1 способ:")
point_estimates_distribution_parameters_1(sample) #пункт 4.1
print("\n2 способ:")
point_estimates_distribution_parameters_2(sample) #пункт 4.2
print("\nДля выборки, увеличенной в 50 раз:")
sample_x50 = np.random.normal(loc=a, scale=np.sqrt(variance), size=50*n)
point_estimates_distribution_parameters_2(sample_x50) #пункт 4.3