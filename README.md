# Expecto

An interactive neural network visualizer. Poke neurons and watch them squirm.

**Live:** https://expecto-eight.vercel.app/

## Features

- Visualize feedforward neural networks in real time on HTML5 Canvas
- Train on built-in datasets: XOR, Circle, Spiral
- Add/remove hidden layers and neurons dynamically
- Switch activation functions (tanh, ReLU, sigmoid)
- Adjust learning rate and training speed
- Drag neurons to rearrange the layout
- Click input neurons to toggle values and propagate activations
- Animated signal propagation along connections
- Color-coded weight visualization (teal = positive, salmon = negative)
- Decision boundary overlay
- Touch-friendly with full mobile support
- Respects `prefers-reduced-motion`

## Tech Stack

- **HTML5 / CSS3 / Vanilla JavaScript** (no framework)
- **[TensorFlow.js](https://www.tensorflow.org/js) v4.17.0** -- in-browser neural network training and inference (loaded from CDN)
- **[Fraunces](https://fonts.google.com/specimen/Fraunces)** -- display serif font (loaded from Google Fonts CDN)
- **[Inter](https://fonts.google.com/specimen/Inter)** -- body sans-serif font (loaded from Google Fonts CDN)

## Dependencies

All dependencies are loaded from CDNs at runtime. There are no npm packages or build tools required.

| Dependency | Version | Source |
|---|---|---|
| TensorFlow.js | 4.17.0 | `cdn.jsdelivr.net` |
| Fraunces | variable | `fonts.gstatic.com` |
| Inter | variable | `fonts.gstatic.com` |

## Getting Started

### Run locally

No build step is needed. Serve the files with any static HTTP server:

```bash
# Using Python
python -m http.server 3000

# Using Node.js (npx, no install required)
npx serve .

# Using PHP
php -S localhost:3000
```

Then open [http://localhost:3000](http://localhost:3000).

### Deploy to Vercel

```bash
npx vercel
```

Or connect the GitHub repository to [vercel.com](https://vercel.com) for automatic deployments. The included `vercel.json` configures the project root as the static output directory.

## Project Structure

```
.
├── index.html        # Entry point
├── script.js         # App logic, neural network, renderer
├── style.css         # All styles
├── vercel.json       # Vercel deployment config
├── LICENSE
└── README.md
```

## License

See [LICENSE](LICENSE).
