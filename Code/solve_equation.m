p=1;c=-2;
sym x
solve('p+2*pi*x+exp(c/(2*x))=0');

x=-10:0.2:10
plot(x, p+2*pi*x+exp(c./(2*x)))
