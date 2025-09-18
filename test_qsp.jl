using LinearAlgebra
using Symbolics
using CairoMakie

phi2phialt(vec) = begin
    res = vec .- 0.5 * pi
    res[1] += 0.25 * pi
    res[end] += 0.25 * pi
    return res
end

S(phi) = [cis(phi) 0; 0 cis(-phi)];

@variables a::Float64 = 1
W = [a 1im*sqrt(1 - a^2); 1im*sqrt(1 - a^2) a];
U = [a sqrt(1 - a^2); sqrt(1 - a^2) -a];


P4(x) = -0.5 + 4x^2 - 3x^4;
P4_phi = [-1.8041560016562335, 0.2617705212477157, 0.4667770770520381, 0.26177052124771505, -0.2333596748613369]
P4_phi = [-0.23343389, 0.26179939, 0.46686777, 0.26179939, -0.23343389]
P4_phialt = phi2phialt(P4_phi)


P2(x) = 2x^2 - 1;
P2_phi = [-0.7903984342278785, 9.547918011776346e-15, 0.7803978925670181]
P2_phialt = phi2phialt(P2_phi)

P4(x) = 2 * x^4 - 1;
P4_phi = [-1.2233962803187972, 0.7785679190069965, -0.6911401458321631, 0.778567919006998, 0.3474000464760987]
P4_phi = [0.3926594, 0.78539816, -0.7853188, 0.78539816, 0.3926594];
P4_phialt = phi2phialt(P4_phi);

Pasin(x) = 0.48047479*x + 0.51952521*x^3
Pasin_phi = [0.10782199, 0.67757591, 0.67757591, 0.10782199]
Pasin_phialt = phi2phialt(Pasin_phi)


# G = map(S, P2_phi);
# H = map(S, P2_phialt);
# Qw = G[1] * W * G[2] * W * G[3];
# Qu = H[1] * U * H[2] * U * H[3];

G = map(S, Pasin_phi);
H = map(S, Pasin_phialt);
Qw =    G[1] * W * G[2] * W * G[3] * W * G[4];
Qu =    H[1] * U * H[2] * U * H[3] * U * H[4];
Qfirst =       U * H[2] * U * H[3] * U * H[4];
Qlast = H[1] * U * H[2] * U * H[3] * U;


x = LinRange(-1, 1, 201);
y = map(Pasin, x);
fw = eval(build_function(Qw[1, 1], a));
pw = map(fw, x);
fu = eval(build_function(Qw[1, 1], a));
pu = map(fu, x);
f_first = eval(build_function(Qfirst[1, 1], a));
p_first = map(f_first, x);
f_last = eval(build_function(Qlast[1, 1], a));
p_last = map(f_last, x);



begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x, asin.(x) * 2/pi, label="asin", linestyle=:dot, color=:black)
    lines!(ax, x, y, label="Exact", linestyle=:solid, color=:black)
    # lines!(ax, x, abs.(pw), label="|QSP|")
    lines!(ax, x, real.(pw), label="Re{QSP}", linestyle=:dash)
    lines!(ax, x, imag.(pw), label="Im{QSP}", linestyle=:dash)
    lines!(ax, x, real.(pu), label="Re{QSVT}", linestyle=:dot, linewidth=4)
    lines!(ax, x, imag.(pu), label="Im{QSVT}", linestyle=:dot, linewidth=4)
    lines!(ax, x, real.(p_first), label="No first", linestyle=:dash)
    lines!(ax, x, real.(p_last), label="No last", linestyle=:dot, linewidth=4)
    axislegend(ax, position=:rb)
    display(fig)
end
