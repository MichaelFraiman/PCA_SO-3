#set page(
  width: 13.333in,
  height: 7.5in,
  margin: (x: 0.9in, y: 0.7in),
)

#set par(justify: false)
#set text(font: "Libertinus Serif", size: 20pt)

#let title-color = rgb("#0F2747")
#let accent = rgb("#2E5EAA")
#let muted = rgb("#666666")

#align(center)[
  #v(12%)

  #text(size: 28pt, weight: "bold", fill: title-color)[
    SO(3)-Invariant PCA
  ]

  #v(8pt)

  #text(size: 28pt, weight: "bold", fill: title-color)[
    with Application to Molecular Data
  ]

  #v(22pt)

  #text(size: 15pt)[
    Michael Fraiman#super[$star.op$] #h(8pt)
    Paulina Hoyos#super[$dagger$] #h(8pt)
    Tamir Bendory#super[$section$] #h(8pt)
    Joe Kileel#super[$dagger$]
  ]
  #text(size: 15pt)[
    Oscar Mickelin#super[$diamond.stroked$] #h(8pt)
    Nir Sharon#super[$star.op$] #h(8pt)
    Amit Singer#super[$ast$]
  ]

  #v(22pt)

  #set text(size: 12pt)
  #super[$star.op$]School of Mathematical Sciences, Tel Aviv University, Israel \
  #super[$dagger$]Department of Mathematics, UT Austin, USA \
  #super[$section$]School of Electrical and Computer Engineering, Tel Aviv University, Israel \
  #super[$diamond.stroked$]Yau Mathematical Sciences Center, Tsinghua University, China \
  #super[$ast$]Program in Applied and Computational Mathematics and Dept. of Mathematics, Princeton University, USA
  #set text(size: 20pt)

]

// ── Slide 2: The Setting ─────────────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[The Setting]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 18pt)

#let light-blue = rgb("#EBF2FB")

#let step-box(number, body) = block(
  width: 100%,
  inset: (x: 12pt, y: 8pt),
  radius: 6pt,
  fill: light-blue,
  stroke: (paint: accent, thickness: 0.7pt),
)[
  #grid(
    columns: (auto, 1fr),
    column-gutter: 10pt,
    align: (horizon, horizon),
    block(
      width: 26pt, height: 26pt,
      radius: 50%,
      fill: accent,
    )[
      #align(center + horizon)[
        #text(size: 14pt, weight: "bold", fill: white)[#number]
      ]
    ],
    body,
  )
]

#step-box("1")[
  #text(weight: "bold")[Data in high dimensions.]
  We have $x_1, dots, x_n in RR^D$ near a low-dimensional affine subspace.
]

#v(1fr)
// Space reserved for image
#align(center)[#text(size: 14pt, style: "italic")[(image)]]
#v(1fr)

#set text(size: 20pt)

// ── Slide 3: The Method ──────────────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[The Method]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(8pt)

#set text(size: 18pt)

#stack(
  dir: ttb,
  spacing: 8pt,

  step-box("2")[
    #text(weight: "bold")[Center & form the covariance operator.]
    $tilde(x)_i = x_i - macron(x)$, #h(6pt)
    $hat(cal(C)) = frac(1,n) sum_i tilde(x)_i tilde(x)_i^top$.
  ],

  step-box("3")[
    #text(weight: "bold")[Diagonalize.]
    $hat(cal(C)) = sum_(k=1)^D lambda_k u_k u_k^top$, #h(4pt)
    $lambda_1 gt.eq dots gt.eq lambda_D gt.eq 0$.
    The $u_k$ are the _principal components_.
  ],

  step-box("4")[
    #text(weight: "bold")[Truncate to rank $d$.]
    $tilde(x)_i approx sum_(k=1)^d chevron.l tilde(x)_i, u_k chevron.r u_k$.
  ],
)

#set text(size: 20pt)

// ── Slide 3: The Bottlenecks ──────────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[The Bottlenecks]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 18pt)

#let light-blue-b = rgb("#EBF2FB")
#let warn-bg = rgb("#FDF0F0")

#let bottleneck-box(number, step-title, step-body, ..problems) = block(
  width: 100%,
  inset: (x: 12pt, y: 10pt),
  radius: 6pt,
  fill: light-blue-b,
  stroke: (paint: accent, thickness: 0.7pt),
)[
  #grid(
    columns: (auto, 1fr),
    column-gutter: 10pt,
    align: (top, top),
    block(
      width: 26pt, height: 26pt,
      radius: 50%,
      fill: accent,
    )[
      #align(center + horizon)[
        #text(size: 14pt, weight: "bold", fill: white)[#number]
      ]
    ],
    [
      #text(weight: "bold")[#step-title ] #step-body
      #for p in problems.pos() [
        #v(5pt)
        #block(
          width: 100%,
          inset: (x: 10pt, y: 8pt),
          radius: 5pt,
          fill: warn-bg,
          stroke: (paint: luma(180), thickness: 0.5pt),
        )[
          #sym.arrow.r.hook #h(4pt) #p
        ]
      ]
    ],
  )
]

#stack(
  dir: ttb,
  spacing: 16pt,

  bottleneck-box("2")[Form the covariance operator.][
    $tilde(x)_i = x_i - macron(x)$, #h(6pt)
    $hat(cal(C)) = frac(1,n) sum_i tilde(x)_i tilde(x)_i^top$.
  ][
    Computing the matrix representation of $hat(cal(C))$ in some arbitrary basis is prohibitive when $D$ is large.
  ][
    Proteins appear in arbitrary orientations, therefore,
      $hat(cal(C))$ must be invariant under $"SO"(3)$.
  ],

  bottleneck-box("3")[Diagonalize the covariance operator.][
    $hat(cal(C)) = sum_(k=1)^D lambda_k u_k u_k^top$.
  ][
    Full diagonalization is prohibitive in high dimensions.
  ],
)

#set text(size: 20pt)

// ── Slide 5: Our Approach ────────────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[Our Approach]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 18pt)

Expand each volume in a _spherical Fourier–Bessel basis_ (spherical Bessel functions $times$ spherical harmonics):

$ phi(r, theta, phi.alt) = sum_(ell = 0)^L sum_(m=-ell)^ell sum_(s=1)^(S(ell)) f_(ell m s) j_(ell s)(r) Y_ell^m (theta, phi.alt) $

#v(4pt)

The $"SO"(3)$-invariant covariance operator:

$
  cal(C) = frac(1, n) sum_(i=1)^n integral_("SO"(3)) (R dot phi^((i)) - phi_"mean") overline((R dot phi^((i)) - phi_"mean")) dif R
$

#v(4pt)

By Wigner $D$-matrix orthogonality, the matrix of $cal(C)$ in the Fourier–Bessel basis is block-diagonal:

$ cal(C) arrow.squiggly C = plus.o.big_(ell=0)^L (I_(2 ell + 1) times.o C_ell), #h(16pt) C_ell (s, s') = frac(1, n) dot frac(1, 2 ell + 1) sum_(i=1)^n sum_(m=-ell)^ell f_(ell m s)^((i)) overline(f_(ell m s')^((i))) $

Each block $C_ell in CC^(S(ell) times S(ell))$ is small and can be diagonalized independently.

#set text(size: 20pt)