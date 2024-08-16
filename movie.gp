q_max = 0.02

set view 70, 35, 1, 1
# unset border
unset xtics
unset ytics
unset ztics
unset key
unset mouse
set palette defined (-q_max "red", 0 "grey", q_max "blue")
set cbrange [-q_max:q_max]
set hidden
# set terminal gif animate delay 2
# set out "panel_c.gif"
do for [i=1:238] {splot "Movie/config.".i.".dat"u 2:3:4:5 with points palette ps 2 pt 7 notitle; pause 0.03}


