%Modified from the R code by Dr. Martin Ridout, which generats a random sample of size n from a
%distribution, given the Laplace transform of its p.d.f.
%http://www.kent.ac.uk/smsas/personal/msr/rlaptrans.html
%
%Mingyuan Zhou
%Oct, 2013
%April, 2014

%#======================================================================================
function rsample =  logBeta_rnd(n, gamma0,cc) %, tol=1e-7, x0=1, xinc=2, m=11, L=1, A=19, nburn=38)
%#======================================================================================
%%{

%     #---------------------------------------------------------
%     # Function for generating a random sample of size n from a
%     # distribution, given the Laplace transform of its p.d.f.
%     #---------------------------------------------------------
tol=1e-7;
x0=1;
xinc=2;
m=11;
L=1;
A=19;
nburn=38;
maxiter = 500;

%       # -----------------------------------------------------
%       # Derived quantities that need only be calculated once,
%       # including the binomial coefficients
%       # -----------------------------------------------------
nterms = nburn + m*L;
seqbtL = nburn:L:nterms;
y = pi * (1i) * (1:nterms) / L;
expy = exp(y);
A2L = 0.5 * A / L;
expxt = exp(A2L) / L;
%coef = bino
%coef = nchoosek(m,0:m) / 2^m;
%coef = gamma(m+1)./gamma(1:m+1)./gamma(m+1:-1:1)/2^m;
coef = exp(gammaln(m+1)-gammaln(1:m+1)-gammaln(m+1:-1:1)-m*log(2));


%       # --------------------------------------------------
%       # Generate sorted uniform random numbers. xrand will
%       # store the corresponding x values
%       # --------------------------------------------------
% u = sort(runif(n), method="qu")
% xrand = u

u = sort(rand(1,n));
xrand = u;

%       #------------------------------------------------------------
%       # Begin by finding an x-value that can act as an upper bound
%       # throughout. This will be stored in upplim. Its value is
%       # based on the maximum value in u. We also use the first
%       # value calculated (along with its pdf and cdf) as a starting
%       # value for finding the solution to F(x) = u_min. (This is
%       # used only once, so doesn't need to be a good starting value
%       #------------------------------------------------------------
t = x0/xinc;
cdf = 0;
kount0 = 0;
set1st = false;
while (kount0 < maxiter && cdf < u(n))
    t = xinc * t;
    kount0 = kount0 + 1;
    x = A2L / t;
    z = x + y/t;
    ltx = ltpdf(x, gamma0,cc);
    ltzexpy = ltpdf(z, gamma0,cc).*expy;
    par.sum = 0.5*real(ltx) + cumsum( real(ltzexpy) );
    par.sum2 = 0.5*real(ltx/x) + cumsum( real(ltzexpy./z) );
    pdf = expxt.*sum(coef.* par.sum(seqbtL)) ./ t;
    cdf = expxt.*sum(coef.* par.sum2(seqbtL)) ./ t;
    if (~set1st && cdf > u(1))
        cdf1 = cdf;
        pdf1 = pdf;
        t1 = t;
        set1st = true;
    end
end
if (kount0 >= maxiter)
    error('Cannot locate upper quantile')
end
upplim = t;
%
%       #--------------------------------
%       # Now use modified Newton-Raphson
%       #--------------------------------

lower = 0;
t = t1;
cdf = cdf1;
pdf = pdf1;
kount = zeros(1,n);

maxiter = 1000;

for j=1:n
     %     #-------------------------------
     %     # Initial bracketing of solution
     %     #-------------------------------

    upper = upplim;
    
    kount(j) = 0;
    while (kount(j) < maxiter && abs(u(j)-cdf) > tol)
        kount(j) = kount(j) + 1;
        %               #-----------------------------------------------
        %               # Update t. Try Newton-Raphson approach. If this
        %               # goes outside the bounds, use midpoint instead
        %               #-----------------------------------------------
        t = t - (cdf-u(j))/pdf;
        if (t < lower || t > upper)
            t = 0.5 .* (lower + upper);
        end
        %
        %               #----------------------------------------------------
        %               # Calculate the cdf and pdf at the updated value of t
        %               #----------------------------------------------------
        x = A2L / t;
        z = x + y/t;
        ltx = ltpdf(x, gamma0,cc);
        ltzexpy = ltpdf(z, gamma0,cc).* expy;
        par.sum = 0.5*real(ltx) + cumsum( real(ltzexpy) );
        par.sum2 = 0.5*real(ltx/x) + cumsum( real(ltzexpy./z) );
        pdf = expxt .* sum(coef .* par.sum(seqbtL)) ./ t;
        cdf = expxt .* sum(coef .* par.sum2(seqbtL)) ./ t;
        
        %               #------------------
        %               # Update the bounds
        %               #------------------
        if (cdf <= u(j))
            lower = t;
        else
            upper = t;
        end
    end
    if (kount(j) >= maxiter)
        disp('Desired accuracy not achieved for F(x)=u')
    end
    xrand(j) = t;
    lower = t;
end

if (n > 1)
    rsample = xrand(randperm(n));
else
    rsample = xrand;
end
%rsample
end


function x = ltpdf(s,gamma0,cc)
%This is the Laplace transform of the logbeta distribution
x = exp(-gamma0*(psi_complex(cc+s)-psi_complex(cc)));
end
