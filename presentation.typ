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

  #text(size: 16pt)[
    #text(weight: "bold")[Speaker:] Michael Fraiman#super[$star.op$]
  ]

  #v(10pt)

  #text(size: 14pt)[
    #text(weight: "bold")[Joint work with:]
    Paulina Hoyos#super[$dagger$] #h(1em)
    Tamir Bendory#super[$section$] #h(1em)
    Joe Kileel#super[$dagger$] #h(1em)
    Oscar Mickelin#super[$diamond.stroked$] #h(1em)
    Nir Sharon#super[$star.op$] #h(1em)
    Amit Singer#super[$ast$] #h(1em)
  ]

  #v(22pt)

  #text(size: 16pt, fill: muted)[
    Presented at IEEE ISBI 2026
  ]

  #v(16pt)

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

#set page(
  numbering: "1",
  number-align: bottom + right,
)

#text(size: 26pt, weight: "bold", fill: title-color)[Setting & Goal]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 15pt)

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

#let note-box(body) = block(
  width: 100%,
  inset: (x: 12pt, y: 8pt),
  radius: 6pt,
  fill: light-blue,
  stroke: (paint: accent, thickness: 0.7pt),
)[
  #body
]

#grid(
  columns: (1fr, 1fr),
  column-gutter: 16pt,
  align: (top, top),
  note-box[
    #text(weight: "bold")[Setting.]
    We observe $n$ three-dimensional functions
    $phi^((1)), dots, phi^((n)): RR^3 -> RR$,
    representing molecular volumes / proteins, each up to an unknown rotation.
  ],
  note-box[
    #text(weight: "bold")[Goal.]
    Find a low-dimensional PCA subspace that approximates the data well,
    but do so in a computationally efficient way that respects rotational symmetry.
  ],
)

#v(6pt)

#align(center)[
  #image("p/PCA.png", width: 55%)
]

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

  step-box("1")[
    #text(weight: "bold")[Data in high dimensions.]
    We have $x_1, dots, x_n in RR^D$ near a low-dimensional affine subspace.
  ],

  step-box("2")[
    #text(weight: "bold")[Center & form a covariance operator.]
    Build an operator that captures the variability of the centered data.
  ],

  step-box("3")[
    #text(weight: "bold")[Diagonalize.]
    Diagonalize the covariance operator.
    Its leading eigenvectors are the _principal components_.
  ],

  step-box("4")[
    #text(weight: "bold")[Truncate to rank $d$.]
    $tilde(x)_i approx sum_(k=1)^d chevron.l tilde(x)_i, u_k chevron.r u_k$.
  ],
)

#set text(size: 20pt)

// ── Slide 3: The Bottlenecks ──────────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[The Problems]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(8pt)

#set text(size: 16pt)

#let light-blue-b = rgb("#EBF2FB")
#let warn-bg = rgb("#FDF0F0")
#let solve-bg = rgb("#EEF8F0")

#let issue-box(body) = block(
  width: 100%,
  inset: (x: 8pt, y: 6pt),
  radius: 5pt,
  fill: warn-bg,
  stroke: (paint: luma(180), thickness: 0.5pt),
)[
  #text(size: 15pt, weight: "bold", fill: rgb("#B42318"))[Issue:] #h(4pt) #body
]

#let solution-box(body) = block(
  width: 100%,
  inset: (x: 8pt, y: 6pt),
  radius: 5pt,
  fill: solve-bg,
  stroke: (paint: rgb("#A6D5AE"), thickness: 0.5pt),
)[
  #text(size: 15pt, weight: "bold", fill: rgb("#166534"))[Solution:] #h(4pt) #body
]

#let bottleneck-box(number, step-title, step-body, ..callouts) = block(
  width: 100%,
  inset: (x: 12pt, y: 6pt),
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
      #for callout in callouts.pos() [
        #v(3pt)
        #callout
      ]
    ],
  )
]

#stack(
  dir: ttb,
  spacing: 8pt,

  bottleneck-box("2")[Form a covariance operator.][
    Construct a covariance operator from the centered data.
  ][
    #issue-box[
      Computing its matrix representation in some arbitrary basis is prohibitive when $D$ is large.
    ]
  ][
    #issue-box[
      Proteins appear in arbitrary orientations, therefore,
      the covariance operator must be invariant under $"SO"(3)$.
    ]
  ][
    #solution-box[
      Average each protein over all rotations when forming the covariance, yielding an $"SO"(3)$-invariant operator.
    ]
  ],

  bottleneck-box("3")[Diagonalize the covariance operator.][
    Extract its leading eigenvectors.
  ][
    #issue-box[
      Full diagonalization is prohibitive in high dimensions.
    ]
  ][
    #solution-box[
      In the spherical Fourier-Bessel basis, the operator is block diagonal, with repeated blocks.
    ]
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


The $"SO"(3)$-invariant *covariance operator*:

$
  cal(C) = frac(1, n) sum_(i=1)^n integral_("SO"(3)) (R dot phi^((i)) - phi_"mean") overline((R dot phi^((i)) - phi_"mean")) dif R
$

#v(4pt)

Expand each volume in a _spherical Fourier–Bessel basis_:

$ phi(r, theta, phi.alt) = sum_(ell = 0)^L sum_(m=-ell)^ell sum_(s=1)^(S(ell)) f_(ell m s) j_(ell s)(r) Y_ell^m (theta, phi.alt) $

#v(4pt)


By Wigner $D$-matrix orthogonality, the matrix of $cal(C)$ in the spherical Fourier–Bessel basis is block-diagonal:

$
  cal(C) arrow.squiggly C = plus.o.big_(ell=0)^L (I_(2 ell + 1) times.o C_ell),
  #h(32pt) C_ell (s, s') = frac(1, n) dot frac(1, 2 ell + 1)
    sum_(i=1)^n sum_(m=-ell)^ell f_(ell m s)^((i)) overline(f_(ell m s')^((i)))
$

Each block $C_ell in CC^(S(ell) times S(ell))$ is small and can be diagonalized independently.

#set text(size: 20pt)

// ── Slide 6: Visual Structure of C ──────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[Visual Structure of $C$]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 18pt)

/*
#v(4pt)

Each local block eigenvector lifts to an eigenvector of $C$ by padding with
zeros.

#v(8pt)
*/
#let c1-fill = rgb("#EAF2FB")
#let c2-fill = rgb("#D7E6F8")
#let c3-fill = rgb("#C3D9F2")
#let empty-fill = rgb("#FCFDFE")

#let matrix-cell(body: [], fill: empty-fill) = block(
  width: 29pt,
  height: 20pt,
  inset: 0pt,
  radius: 3pt,
  fill: fill,
  stroke: (paint: luma(220), thickness: 0.55pt),
)[
  #align(center + horizon)[#body]
]

#let z = matrix-cell()
#let b1 = matrix-cell(body: [$C_0$], fill: c1-fill)
#let b2 = matrix-cell(body: [$C_1$], fill: c2-fill)
#let b3 = matrix-cell(body: [$C_2$], fill: c3-fill)
#let mdots = matrix-cell(body: [#text(size: 12pt, fill: muted)[...]])

#let vector-cell(body: [], fill: empty-fill) = block(
  width: 29pt,
  height: 18pt,
  inset: 0pt,
  radius: 2pt,
  fill: fill,
  stroke: (paint: luma(220), thickness: 0.4pt),
)[
  #set text(size: 10pt)
  #align(center + horizon)[#body]
]

#let vz = vector-cell()
#let u0 = vector-cell(body: [$u_0$], fill: c1-fill)
#let u1 = vector-cell(body: [$u_1$], fill: c2-fill)
#let u2 = vector-cell(body: [$u_2$], fill: c3-fill)
#let vdots = vector-cell(body: [#text(size: 11pt, fill: muted)[...]])

#let vector-box(..cells) = grid(
  columns: 1fr,
  row-gutter: 2.5pt,
  align: (horizon, horizon),
  ..cells.pos(),
)

#let v0 = vector-box(u0, vz, vz, vz, vz, vz, vz, vz, vz)
#let v1a = vector-box(vz, u1, vz, vz, vz, vz, vz, vz, vz)
#let v1b = vector-box(vz, vz, u1, vz, vz, vz, vz, vz, vz)
#let v1c = vector-box(vz, vz, vz, u1, vz, vz, vz, vz, vz)
#let v2a = vector-box(vz, vz, vz, vz, u2, vz, vz, vz, vz)
#let v2b = vector-box(vz, vz, vz, vz, vz, u2, vz, vz, vz)
#let v2c = vector-box(vz, vz, vz, vz, vz, vz, u2, vz, vz)
#let v2d = vector-box(vz, vz, vz, vz, vz, vz, vz, u2, vz)
#let v2e = vector-box(vz, vz, vz, vz, vz, vz, vz, vz, u2)
#let vmore = vector-box(vz, vz, vz, vz, vdots, vz, vz, vz, vz)

#align(horizon + center)[
  #grid(
    columns: (auto, auto),
    column-gutter: 18pt,
    align: (horizon, horizon,),
    [$ C = $],
    block(
      inset: 8pt,
      radius: 8pt,
      fill: white,
      stroke: (paint: accent, thickness: 0.8pt),
    )[
      #grid(
        columns: 10,
        column-gutter: 2.5pt,
        row-gutter: 2.5pt,

        b1, z, z, z, z, z, z, z, z, z,
        z, b2, z, z, z, z, z, z, z, z,
        z, z, b2, z, z, z, z, z, z, z,
        z, z, z, b2, z, z, z, z, z, z,
        z, z, z, z, b3, z, z, z, z, z,
        z, z, z, z, z, b3, z, z, z, z,
        z, z, z, z, z, z, b3, z, z, z,
        z, z, z, z, z, z, z, b3, z, z,
        z, z, z, z, z, z, z, z, b3, z,
        z, z, z, z, z, z, z, z, z, mdots,
      )
    ],/*
    [
      #set text(size: 18pt)
      #text(weight: "bold")[Eigenvectors]
      #v(6pt)
      #grid(
        columns: (1fr,1fr,1fr,1fr,1fr,1fr,1fr,1fr,1fr,1fr),
        column-gutter: 4pt,
        align: (top, top),
        v0, v1a, v1b, v1c, v2a, v2b, v2c, v2d, v2e, vmore,
      )
    ],*/
  )
]

#set text(size: 20pt)

// ── Slide 7: Computational Complexity ───────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[Computational Complexity]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(12pt)

#set text(size: 17pt)

#text(fill: black)[
  $N$ — grid side length, #h(10pt)
  $L$ — angular bandlimit, #h(10pt)
  $S = max_ell S(ell)$ — maximal number of radial terms.
]

#v(10pt)

#let complexity-box(title, body) = block(
  width: 100%,
  inset: (x: 12pt, y: 10pt),
  radius: 6pt,
  fill: white,
  stroke: (paint: accent, thickness: 0.7pt),
)[
  #text(weight: "bold", fill: title-color)[#title]
  #v(6pt)
  #body
]

#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 16pt,
  align: (top, top),

  complexity-box("Expansion")[
    SFB expansion:
    #v(6pt)
    #text(size: 20pt, weight: "bold", fill: accent)[$O(N^3 log^2 N)$]
  ],

  complexity-box("Invariant PCA (out method)")[
    Block covariances + block eigendecompositions:
    #v(6pt)
    #text(size: 18pt, weight: "bold", fill: accent)[$O(S^2 L^2 + S^3 L)$]
  ],


  complexity-box("Naive Computation")[
    Full dense covariance + eigendecomposition:
    #v(6pt)
    #text(size: 18pt, weight: "bold", fill: accent)[$O(S^2 L^4) + O(S^3 L^6)$]
  ],
)

#v(10pt)

#block(
  width: 100%,
  inset: (x: 14pt, y: 10pt),
  radius: 6pt,
  fill: light-blue,
  stroke: (paint: accent, thickness: 0.8pt),
)[
  #text(weight: "bold", fill: title-color)[Nyquist Regime]
  #v(4pt)
  For covariance computation, with $S = Theta(N)$ and $L = O(N)$, #h(8pt)
  #text(size: 20pt, weight: "bold", fill: accent)[$O(N^4)$ vs. $O(N^9)$]
]

#set text(size: 20pt)

// ── Slide 8: Numerical Results ───────────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[Numerical Results]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(8pt)

#set text(size: 15pt)

#let pca-color = rgb("#1f77b4")
#let bh-sort-color = rgb("#ff7f0e")
#let bh-root-color = rgb("#2ca02c")

#align(left)[
  #block(width: 100%)[
    #align(left)[
      #text(size: 18pt)[
        For the expansion $phi = sum_(j=1)^D alpha_j v_j$ in an orthonormal basis
        $V = (v_1, dots, v_D)$, we compute:
      ]
    ]
  ]
]

#v(4pt)

#align(center)[
  #block(
    width: 36%,
    inset: (x: 12pt, y: 6pt),
    radius: 6pt,
    fill: light-blue,
    stroke: (paint: accent, thickness: 0.7pt),
  )[
    #align(center)[$
      w_phi^V (d) = frac(sum_(j=1)^d |alpha_j|^2, sum_(j=1)^D |alpha_j|^2)
    $]
  ]
]

#v(4pt)

#let result-plot(title, path) = block(
  width: 100%,
  breakable: false,
  inset: 0mm,
)[
  #align(center)[
    #text(size: 14pt, weight: "bold", fill: title-color)[#title]
  ]
  #align(center)[
    #image(path, width: 64%)
  ]
]

#grid(
  columns: (1fr, 1fr),
  column-gutter: 12pt,
  inset: 0mm,
  row-gutter: 0mm,
  align: (top, top),
  result-plot([$N = 64$], "paper/p/1fzf_pca_vs_fb_N=64_first100.png"),
  result-plot([$N = 256$], "paper/p/1fzf_pca_vs_fb_N=256_first100.png"),
)

#align(center)[
  #block(width: 94%)[
    #align(center)[
      #text(size: 11pt)[
        Sample volume: #text(weight: "bold")[`1fzf`].
        Curves:
        #text(weight: "bold", fill: pca-color)[PCA], #h(6pt)
        #text(weight: "bold", fill: bh-sort-color)[SFB sorted by coefficient magnitude], #h(6pt)
        #text(weight: "bold", fill: bh-root-color)[SFB sorted by increasing $u_(ell s)$
        ($s$-th positive root of $j_ell$)].
      ]
    ]
  ]
]

#set text(size: 20pt)

// ── Slide 9: Volume Reconstructions ──────────────────────────────────────────
#pagebreak()

#text(size: 26pt, weight: "bold", fill: title-color)[Volume Reconstructions]
#v(4pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1pt))
#v(8pt)

#set text(size: 18pt)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  column-gutter: 8pt,
  row-gutter: 8pt,
  align: (top, top),

  [#align(center)[#text(weight: "bold", fill: title-color)[Reference]]],
  [#align(center)[#text(weight: "bold", fill: title-color)[SFB]]],
  [#align(center)[#text(weight: "bold", fill: title-color)[$d = 10$]]],
  [#align(center)[#text(weight: "bold", fill: title-color)[$d = 20$]]],
  [#align(center)[#text(weight: "bold", fill: title-color)[$d = 100$]]],

  [#align(center)[#image("paper/p/pdb1.png", width: 82%)]],
  [#align(center)[#image("paper/p/exp1.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap1-10.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap1-20.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap1-100.png", width: 82%)]],

  [#align(center)[#image("paper/p/pdb2.png", width: 82%)]],
  [#align(center)[#image("paper/p/exp2.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap2-10.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap2-20.png", width: 82%)]],
  [#align(center)[#image("paper/p/ap2-100.png", width: 82%)]],
)

#v(6pt)

#align(center)[
  #block(width: 95%)[
    #align(center)[
      #text(size: 16pt)[
        $N = 128$. Top row: #text(weight: "bold")[`1avo`]. Bottom row:
        #text(weight: "bold")[`1dgb`]. Left to right: reference volume,
        spherical Fourier–Bessel expansion, and PCA reconstructions with #box[$d = 10, 20, 100$.]
      ]
    ]
  ]
]

#set text(size: 20pt)

// ── Slide 10: Thank You ─────────────────────────────────────────────────────
#pagebreak()

#set page(numbering: none)

#align(center + horizon)[
  
   #text(size: 38pt, weight: "bold", fill: title-color)[
    Thank you!
  ]
]
