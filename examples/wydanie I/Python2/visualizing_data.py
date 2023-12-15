# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from collections import Counter

def make_chart_simple_line_chart(plt):

    years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

    # Utwórz wykres liniowy, przedstaw lata na osi x, a PKB na osi y.
    plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

    # Dodaj tytuł.
    plt.title("Nominalny PKB")

    # Dodaj tytuł osi y.
    plt.ylabel("Mld dol.")
    plt.show()


def make_chart_simple_bar_chart(plt):

    movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
    num_oscars = [5, 11, 3, 8, 10]

    xs = [i for i, _ in enumerate(movies)]

    # Narysuj słupki o współrzędnych x [xs] i wysokości [num_oscars].
    plt.bar(xs, num_oscars)
    plt.ylabel("Liczba nagrod")
    plt.title("Moje ulubione filmy")

    # Na osi x nanieś etykietę w postaci tytułów filmów i je wyśrodkuj.
    plt.xticks([i for i, _ in enumerate(movies)], movies)
    
    plt.show()

def make_chart_histogram(plt):
    grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
    decile = lambda grade: grade // 10 * 10 
    histogram = Counter(decile(grade) for grade in grades)

    plt.bar([x for x in histogram.keys()],     # Wyśrodkuj śłupki.
            histogram.values(),                # Nadaj każdemu słupkowi właściwą wysokość.
            8)                                 # Nadaj każdemu słupkowi szerokość 8.
    plt.axis([-5, 105, 0, 5])                  # Zdefiniuj zakres osi x od -5 do 105,
                                               # i zakres osi y od 0 do 5.
    plt.xticks([10 * i for i in range(11)])    # Umieść etykiety osi x w punktach 0, 10, ..., 100.
    plt.xlabel("Decyl")
    plt.ylabel("Liczba studentow")
    plt.title("Rozklad ocen z pierwszego egzaminu")
    plt.show()

def make_chart_misleading_y_axis(plt, mislead=True):

    mentions = [500, 505]
    years = [2013, 2014]

    plt.bar([2013, 2014], mentions, 0.8)
    plt.xticks(years)
    plt.ylabel("Ile razy uslyszalem fraze nauka o danych?")

    # Jeżeli tego nie zrobisz, to matplotlib umieści na osi x etykiety 0 i 1
    # a następnie doda w rogu +2.013e3 (niegrzeczny matplotlib!).
    plt.ticklabel_format(useOffset=False)

    if mislead:
        # W celu zamazania obrazu sytuacji oś y zaczyna się od punktu 500.
        plt.axis([2012.5,2014.5,499,506])
        plt.title("Patrz na ten ogromny wzrost!")
    else:
        plt.axis([2012.5,2014.5,0,550])
        plt.title("Niewielki wzrost")       
    plt.show()

def make_chart_several_line_charts(plt):

    variance     = [1,2,4,8,16,32,64,128,256]
    bias_squared = [256,128,64,32,16,8,4,2,1]
    total_error  = [x + y for x, y in zip(variance, bias_squared)]

    xs = range(len(variance))

    # Funkcja plt.plot może być wywoływana wielokrotnie
    # w celu umieszczenia wielu serii danych na tym samym wykresie.
    plt.plot(xs, variance,     'g-',  label='wariancja')    # zielona linia ciągła
    plt.plot(xs, bias_squared, 'r-.', label='prog^2')      # czerwona linia składająca się z kropek i kresek
    plt.plot(xs, total_error,  'b:',  label='blad calkowity') # niebieska linia punktowana

    # Przypisaliśmy etykiety do wszystkich serii danych,
    # a więc legenda zostanie wygenerowana automatycznie.
    # Parametr loc=9 umieszcza ją na środku u góry wykresu.
    plt.legend(loc=9)
    plt.xlabel("Stopien skomplikowania modelu")
    plt.title("Kompromis pomiędzy wartoscia progowa i zlozonoscia modelu")
    plt.show()

def make_chart_scatter_plot(plt):

    friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
    minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    plt.scatter(friends, minutes)
    
    # Nadaj etykiety wszystkim punktom.
    for label, friend_count, minute_count in zip(labels, friends, minutes):
        plt.annotate(label,
                     xy=(friend_count, minute_count), # Umieść etykietę we właściwym miejscu,
                     xytext=(5, -5), # ale lekko ją przesuń.
                     textcoords='offset points')

    plt.title("Czas spedzony na stronie a liczba znajomych")
    plt.xlabel("Liczba znajomych")
    plt.ylabel("Dzienny czas spedzony na stronie (minuty)")
    plt.show()

def make_chart_scatterplot_axes(plt, equal_axes=False):

    test_1_grades = [ 99, 90, 85, 97, 80]
    test_2_grades = [100, 85, 60, 90, 70]

    plt.scatter(test_1_grades, test_2_grades)
    plt.xlabel("test 1")
    plt.ylabel("test 2")

    if equal_axes:
        plt.title("Osie sa porownywalne")
        plt.axis("equal")
    else:
        plt.title("Osie nie sa porownywalne")

    plt.show()

def make_chart_pie_chart(plt):

    plt.pie([0.95, 0.05], labels=["Uses pie charts", "Knows better"])

    # make sure pie is a circle and not an oval
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":

    make_chart_simple_line_chart(plt)

    make_chart_simple_bar_chart(plt)

    make_chart_histogram(plt)

    make_chart_misleading_y_axis(plt, mislead=True)

    make_chart_misleading_y_axis(plt, mislead=False)

    make_chart_several_line_charts(plt)

    make_chart_scatterplot_axes(plt, equal_axes=False)

    make_chart_scatterplot_axes(plt, equal_axes=True)

    make_chart_pie_chart(plt)
