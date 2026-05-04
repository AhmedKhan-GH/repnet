# Butler et al. ECG-AI Model Architecture (Shelved)

## Subsection Text (from Convolutional Transformer Progress section)

### Butler et al. ECG-AI Model Architecture

The Butler et al. ECG-AI model is a 1-D residual convolutional neural network that accepts 12-lead ECG signals of length 2250 samples. The architecture comprises six residual blocks organized into three stages of progressively increasing filter width (16, 32, 64), each followed by max-pooling and dropout for regularization. A final flatten and dense softmax layer produces a two-class output. See Appendix~B for the full decomposed layer-by-layer model architecture.

#### High-Level TikZ Figure

```latex
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=0.35cm,
    block/.style={draw, rounded corners, minimum height=0.55cm, minimum width=4.5cm,
                  font=\scriptsize\sffamily, align=center, thick},
    stage16/.style={block, fill=blue!15},
    stage32/.style={block, fill=orange!20},
    stage64/.style={block, fill=red!15},
    io/.style={block, fill=green!15},
    arrow/.style={-{Stealth[length=2pt]}, thick},
    stagelabel/.style={font=\tiny\sffamily\bfseries, text=gray},
]

\node[io] (input) {Input: $2250\times12$};
\node[stage16, below=of input] (rb1) {ResBlock $16\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[stage16, below=of rb1] (rb2) {ResBlock $16\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[stage32, below=0.6cm of rb2] (rb3) {ResBlock $32\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[stage32, below=of rb3] (rb4) {ResBlock $32\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[stage64, below=0.6cm of rb4] (rb5) {ResBlock $64\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[stage64, below=of rb5] (rb6) {ResBlock $64\times3$ $\to$ MaxPool $\to$ Drop 0.1};
\node[io, below=0.6cm of rb6] (flat) {Flatten $\to$ Dense(2) $\to$ Softmax};

\foreach \a/\b in {input/rb1, rb1/rb2, rb2/rb3, rb3/rb4, rb4/rb5, rb5/rb6, rb6/flat}{
    \draw[arrow] (\a) -- (\b);
}

\begin{scope}[on background layer]
    \node[fit=(rb1)(rb2), inner sep=5pt, draw=blue!40, fill=blue!5, rounded corners, dashed] (s1) {};
    \node[stagelabel, right=3pt of s1] {Stage 1 -- 16 filters};
    \node[fit=(rb3)(rb4), inner sep=5pt, draw=orange!50, fill=orange!5, rounded corners, dashed] (s2) {};
    \node[stagelabel, right=3pt of s2] {Stage 2 -- 32 filters};
    \node[fit=(rb5)(rb6), inner sep=5pt, draw=red!40, fill=red!5, rounded corners, dashed] (s3) {};
    \node[stagelabel, right=3pt of s3] {Stage 3 -- 64 filters};
\end{scope}

\end{tikzpicture}
\caption{High-level view of the Butler et al.\ ECG-AI architecture showing six ResBlocks across three filter-width stages (16, 32, 64), each stage followed by max-pooling and dropout, culminating in a dense softmax output layer.}
\label{fig:butler-arch}
\end{figure}
```

---

## Appendix: Full Layer-by-Layer Architecture

```latex
\clearpage
\section{Butler et al. Full Layer-by-Layer Architecture}
\label{appendix:butler}
\vspace{1em}
\noindent\centering
\resizebox{!}{0.75\textheight}{%
\begin{tikzpicture}[
    node distance=0.15cm,
    layer/.style={draw, rounded corners=2pt, minimum height=0.38cm, minimum width=1.8cm,
                  font=\tiny\sffamily, align=center, thick},
    conv/.style={layer, fill=blue!12},
    bn/.style={layer, fill=yellow!15},
    act/.style={layer, fill=green!12},
    add/.style={layer, fill=purple!12},
    pool/.style={layer, fill=gray!15},
    drop/.style={layer, fill=red!8},
    head/.style={layer, fill=green!20},
    io/.style={layer, fill=teal!15},
    proj/.style={layer, fill=orange!15},
    arrow/.style={-{Stealth[length=1.5pt]}, semithick},
    skiparrow/.style={-{Stealth[length=1.5pt]}, semithick, dashed, gray},
    stagelabel/.style={font=\tiny\sffamily\bfseries, text=gray},
]

% ---- Column 1: Blocks 1-2 (16 filters) ----
\node[io] (in) {Input $2250\times12$};

% Block 1
\node[conv, below=0.3cm of in] (c0) {Conv1D 16, k=3};
\node[bn, below=of c0] (b0) {BatchNorm};
\node[act, below=of b0] (a0) {LeakyReLU};
\node[conv, below=of a0] (c1) {Conv1D 16, k=3};
\node[bn, below=of c1] (b1) {BatchNorm};
\node[act, below=of b1] (a1) {LeakyReLU};
\node[conv, below=of a1] (c2) {Conv1D 16, k=3};
\node[bn, below=of c2] (b2) {BatchNorm};
\node[add, below=of b2] (add0) {Add};
\node[act, below=of add0] (a2) {LeakyReLU};
\node[pool, below=of a2] (mp0) {MaxPool1D};
\node[drop, below=of mp0] (d0) {Dropout 0.1};

% Block 2
\node[conv, below=0.45cm of d0] (c3) {Conv1D 16, k=3};
\node[bn, below=of c3] (b3) {BatchNorm};
\node[act, below=of b3] (a3) {LeakyReLU};
\node[conv, below=of a3] (c4) {Conv1D 16, k=3};
\node[bn, below=of c4] (b4) {BatchNorm};
\node[add, below=of b4] (add1) {Add};
\node[act, below=of add1] (a4) {LeakyReLU};
\node[pool, below=of a4] (mp1) {MaxPool1D};
\node[drop, below=of mp1] (d1) {Dropout 0.1};

% ---- Column 2: Blocks 3-4 (32 filters) ----
\node[conv, right=2.5cm of in] (c5) {Conv1D 32, k=3};
\node[bn, below=of c5] (b5) {BatchNorm};
\node[act, below=of b5] (a5) {LeakyReLU};
\node[conv, below=of a5] (c6) {Conv1D 32, k=3};
\node[proj, below=of c6] (c7) {Conv1D 32, k=1};
\node[bn, below=of c7] (b6) {BatchNorm};
\node[add, below=of b6] (add2) {Add};
\node[act, below=of add2] (a6) {LeakyReLU};
\node[pool, below=of a6] (mp2) {MaxPool1D};
\node[drop, below=of mp2] (d2) {Dropout 0.1};

% Block 4
\node[conv, below=0.45cm of d2] (c8) {Conv1D 32, k=3};
\node[bn, below=of c8] (b7) {BatchNorm};
\node[act, below=of b7] (a7) {LeakyReLU};
\node[conv, below=of a7] (c9) {Conv1D 32, k=3};
\node[bn, below=of c9] (b8) {BatchNorm};
\node[add, below=of b8] (add3) {Add};
\node[act, below=of add3] (a8) {LeakyReLU};
\node[pool, below=of a8] (mp3) {MaxPool1D};
\node[drop, below=of mp3] (d3) {Dropout 0.1};

% ---- Column 3: Blocks 5-6 (64 filters) ----
\node[conv, right=2.5cm of c5] (c10) {Conv1D 64, k=3};
\node[bn, below=of c10] (b9) {BatchNorm};
\node[act, below=of b9] (a9) {LeakyReLU};
\node[conv, below=of a9] (c11) {Conv1D 64, k=3};
\node[proj, below=of c11] (c12) {Conv1D 64, k=1};
\node[bn, below=of c12] (b10) {BatchNorm};
\node[add, below=of b10] (add4) {Add};
\node[act, below=of add4] (a10) {LeakyReLU};
\node[pool, below=of a10] (mp4) {MaxPool1D};
\node[drop, below=of mp4] (d4) {Dropout 0.1};

% Block 6
\node[conv, below=0.45cm of d4] (c13) {Conv1D 64, k=3};
\node[bn, below=of c13] (b11) {BatchNorm};
\node[act, below=of b11] (a11) {LeakyReLU};
\node[conv, below=of a11] (c14) {Conv1D 64, k=3};
\node[bn, below=of c14] (b12) {BatchNorm};
\node[add, below=of b12] (add5) {Add};
\node[act, below=of add5] (a12) {LeakyReLU};
\node[pool, below=of a12] (mp5) {MaxPool1D};
\node[drop, below=of mp5] (d5) {Dropout 0.1};

% Classification head — separate from stages
\node[head, below=0.45cm of d5] (flat) {Flatten};
\node[head, below=of flat] (dense) {Dense(2)};
\node[head, below=of dense] (softmax) {Softmax};

% ---- Sequential arrows within columns ----
% Col 1
\foreach \a/\b in {in/c0, c0/b0, b0/a0, a0/c1, c1/b1, b1/a1, a1/c2, c2/b2, b2/add0, add0/a2, a2/mp0, mp0/d0,
                    d0/c3, c3/b3, b3/a3, a3/c4, c4/b4, b4/add1, add1/a4, a4/mp1, mp1/d1}{
    \draw[arrow] (\a) -- (\b);
}
% Col 2
\foreach \a/\b in {c5/b5, b5/a5, a5/c6, c6/c7, c7/b6, b6/add2, add2/a6, a6/mp2, mp2/d2,
                    d2/c8, c8/b7, b7/a7, a7/c9, c9/b8, b8/add3, add3/a8, a8/mp3, mp3/d3}{
    \draw[arrow] (\a) -- (\b);
}
% Col 3 + head
\foreach \a/\b in {c10/b9, b9/a9, a9/c11, c11/c12, c12/b10, b10/add4, add4/a10, a10/mp4, mp4/d4,
                    d4/c13, c13/b11, b11/a11, a11/c14, c14/b12, b12/add5, add5/a12, a12/mp5, mp5/d5,
                    d5/flat, flat/dense, dense/softmax}{
    \draw[arrow] (\a) -- (\b);
}

% Cross-column arrows — rectangular path between columns
\coordinate (mid12) at ($(d1.east)!0.5!(c5.west)$);
\coordinate (mid23) at ($(d3.east)!0.5!(c10.west)$);
\draw[arrow] (d1.east) -- (mid12 |- d1.east) |- (c5.west);
\draw[arrow] (d3.east) -- (mid23 |- d3.east) |- (c10.west);

% Skip connections (dashed) — routed to the right of each column
\draw[skiparrow] (a0.east) -- ++(0.35,0) |- (add0.east);
\draw[skiparrow] (d0.east) -- ++(0.35,0) |- (add1.east);
\draw[skiparrow] (c5.east) -- ++(0.35,0) |- (add2.east);
\draw[skiparrow] (d2.east) -- ++(0.35,0) |- (add3.east);
\draw[skiparrow] (c10.east) -- ++(0.35,0) |- (add4.east);
\draw[skiparrow] (d4.east) -- ++(0.35,0) |- (add5.east);

% Stage labels at top
\begin{scope}[on background layer]
    % Stage 1 — two res blocks
    \node[fit=(c0)(d0), inner sep=4pt, draw=blue!40, fill=blue!5, rounded corners, dashed] (s1a) {};
    \node[stagelabel, left=2pt of s1a] {Block 1};
    \node[fit=(c3)(d1), inner sep=4pt, draw=blue!40, fill=blue!5, rounded corners, dashed] (s1b) {};
    \node[stagelabel, left=2pt of s1b] {Block 2};
    % Stage 2 — two res blocks
    \node[fit=(c5)(d2), inner sep=4pt, draw=orange!50, fill=orange!5, rounded corners, dashed] (s2a) {};
    \node[stagelabel, left=2pt of s2a] {Block 3};
    \node[fit=(c8)(d3), inner sep=4pt, draw=orange!50, fill=orange!5, rounded corners, dashed] (s2b) {};
    \node[stagelabel, left=2pt of s2b] {Block 4};
    % Stage 3 — two res blocks
    \node[fit=(c10)(d4), inner sep=4pt, draw=red!40, fill=red!5, rounded corners, dashed] (s3a) {};
    \node[stagelabel, left=2pt of s3a] {Block 5};
    \node[fit=(c13)(d5), inner sep=4pt, draw=red!40, fill=red!5, rounded corners, dashed] (s3b) {};
    \node[stagelabel, left=2pt of s3b] {Block 6};
    % Stage group labels — all at same vertical position
    \node[stagelabel, above=3pt of s2a] (s2label) {Stage 2 -- 32 filters};
    \node[stagelabel, above=3pt of s3a] {Stage 3 -- 64 filters};
    \node[stagelabel] at (s1a.north |- s2label) {Stage 1 -- 16 filters};
    \node[fit=(flat)(softmax), inner sep=5pt, draw=green!50, fill=green!3, rounded corners, dashed] (sh) {};
    \node[stagelabel, right=3pt of sh] {Head};
\end{scope}

\end{tikzpicture}%
}
\captionof{figure}{Complete layer-by-layer architecture of the Butler et al.\ ECG-AI model, expanded from Figure~\ref{fig:butler-arch}. The network comprises six residual blocks across three stages of increasing filter width: Stage~1 (16 filters), Stage~2 (32 filters), and Stage~3 (64 filters). Each layer is shown individually --- Conv1D (blue), BatchNorm (yellow), LeakyReLU (green), Add (purple), MaxPool1D (gray), and Dropout (red). Dashed gray arrows indicate residual skip connections that bypass the convolutional layers within each block. Orange blocks denote $1\times1$ projection convolutions at Blocks~3 and~5, where the skip path requires dimensionality matching due to the increase in filter count. The classification head (Flatten, Dense, Softmax) follows after Stage~3.}
\label{fig:butler-detail}
```
