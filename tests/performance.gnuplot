#!/usr/bin/env gnuplot
set terminal svg size 1024,768
set output "performance.svg"
set boxwidth .9
set style fill solid
set xtics rotate by -80

# pro: get to see smaller times easier
# con: stacked bar plots no longer represent visually the proportions of their time
#set log y

plot "performance.txt" using 4:xtic(1) with boxes title "total",\
	'' using 3:xtic(1) with boxes title "feed-forward + backpropgate",\
	'' using 2:xtic(1) with boxes title "feed-forward"
