(() => {
  "use strict";

  const COLORS = {
    positive: "#6f9d9a",
    negative: "#d98b73",
    accent: "#d6b347",
    accentGlow: "rgba(214, 179, 71, 0.3)",
    neuronFill: "#292524",
    neuronStroke: "#d6b347",
    connectionBase: "rgba(70, 63, 58, 0.25)",
    signal: "#d6b347",
    text: "#e8e0d4",
    textMuted: "#7c736c",
    grid: "rgba(70, 63, 58, 0.08)",
    mustard: "#d6b347",
    teal: "#6f9d9a",
    salmon: "#d98b73",
    slime: "#9fb06f",
    lavender: "#b8a7cc",
    border: "#463f3a",
    bgPrimary: "#1a1816",
    bgTertiary: "#292524",
  };

  const LAYER_SPACING_RATIO = 0.18;
  const NEURON_BASE_RADIUS = 18;
  const SIGNAL_SPEED = 0.02;

  class DataGenerator {
    static xor(n = 200) {
      const data = [];
      const labels = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        data.push([x, y]);
        labels.push(((x > 0) ^ (y > 0)) ? 1 : 0);
      }
      return { data, labels, inputDim: 2 };
    }

    static circle(n = 200) {
      const data = [];
      const labels = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        data.push([x, y]);
        labels.push(x * x + y * y < 0.5 ? 1 : 0);
      }
      return { data, labels, inputDim: 2 };
    }

    static spiral(n = 300) {
      const data = [];
      const labels = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        const t = (i / half) * 4 * Math.PI;
        const r = t / (4 * Math.PI) * 0.8;
        const x = r * Math.cos(t) + (Math.random() - 0.5) * 0.1;
        const y = r * Math.sin(t) + (Math.random() - 0.5) * 0.1;
        data.push([x, y]);
        labels.push(0);
      }
      for (let i = 0; i < half; i++) {
        const t = (i / half) * 4 * Math.PI + Math.PI;
        const r = t / (4 * Math.PI) * 0.8;
        const x = r * Math.cos(t) + (Math.random() - 0.5) * 0.1;
        const y = r * Math.sin(t) + (Math.random() - 0.5) * 0.1;
        data.push([x, y]);
        labels.push(1);
      }
      return { data, labels, inputDim: 2 };
    }
  }

  class NeuralNetwork {
    constructor() {
      this.model = null;
      this.architecture = [2, 4, 4, 1];
      this.activation = "tanh";
      this.weights = [];
      this.activations = [];
      this.epoch = 0;
      this.loss = null;
      this.buildModel();
    }

    buildModel() {
      if (this.model) {
        this.model.dispose();
        tf.dispose(this.weights);
      }

      const layers = [];
      for (let i = 0; i < this.architecture.length - 1; i++) {
        const units = this.architecture[i + 1];
        const isLast = i === this.architecture.length - 2;
        layers.push(
          tf.layers.dense({
            inputShape: i === 0 ? [this.architecture[0]] : undefined,
            units,
            activation: isLast ? "sigmoid" : this.activation,
            kernelInitializer: "glorotUniform",
            biasInitializer: "zeros",
          })
        );
      }

      this.model = tf.sequential({ layers });
      this.model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "binaryCrossentropy",
      });

      this.extractWeights();
      this.epoch = 0;
      this.loss = null;
    }

    extractWeights() {
      tf.tidy(() => {
        this.weights = this.model.layers.map((layer) => {
          const w = layer.getWeights();
          if (w.length === 0) return null;
          return {
            kernel: w[0].arraySync(),
            bias: w[1].arraySync(),
          };
        }).filter(Boolean);
      });
    }

    async trainStep(dataT, labelsT) {
      const lossTensor = await this.model.trainOnBatch(dataT, labelsT);
      const lossVal = lossTensor.dataSync()[0];
      lossTensor.dispose();
      this.extractWeights();
      return lossVal;
    }

    predict(inputs) {
      return tf.tidy(() => {
        const inputT = tf.tensor2d(inputs);
        const output = this.model.predict(inputT);
        return output.arraySync();
      });
    }

    getActivations(input) {
      return tf.tidy(() => {
        let current = tf.tensor2d([input]);
        const activations = [current.arraySync()[0]];

        for (const layer of this.model.layers) {
          current = layer.apply(current);
          activations.push(current.arraySync()[0]);
        }

        return activations;
      });
    }

    addLayer(neurons = 4) {
      this.architecture.splice(this.architecture.length - 1, 0, neurons);
      this.buildModel();
    }

    removeLayer() {
      if (this.architecture.length <= 3) return;
      this.architecture.splice(this.architecture.length - 2, 1);
      this.buildModel();
    }

    addNeuron() {
      for (let i = 1; i < this.architecture.length - 1; i++) {
        this.architecture[i] = Math.min(this.architecture[i] + 1, 12);
      }
      this.buildModel();
    }

    removeNeuron() {
      for (let i = 1; i < this.architecture.length - 1; i++) {
        this.architecture[i] = Math.max(this.architecture[i] - 1, 1);
      }
      this.buildModel();
    }

    reset() {
      this.buildModel();
    }

    getParamCount() {
      let count = 0;
      for (let i = 0; i < this.architecture.length - 1; i++) {
        const input = this.architecture[i];
        const output = this.architecture[i + 1];
        count += input * output + output;
      }
      return count;
    }
  }

  class Renderer {
    constructor(canvas, boundaryCanvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.boundaryCanvas = boundaryCanvas;
      this.boundaryCtx = boundaryCanvas.getContext("2d");
      this.neurons = [];
      this.signals = [];
      this.time = 0;
      this.showWeights = true;
      this.showSignals = true;
      this.showBoundary = false;
      this.sampleInput = [0.5, 0.5];
      this.resize();
    }

    resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.canvas.style.width = rect.width + "px";
      this.canvas.style.height = rect.height + "px";
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      this.boundaryCanvas.width = rect.width * dpr;
      this.boundaryCanvas.height = rect.height * dpr;
      this.boundaryCanvas.style.width = rect.width + "px";
      this.boundaryCanvas.style.height = rect.height + "px";
      this.boundaryCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      this.width = rect.width;
      this.height = rect.height;
    }

    layoutNetwork(architecture, existingPositions) {
      const layerCount = architecture.length;
      const spacing = this.width * LAYER_SPACING_RATIO;
      const startX = (this.width - spacing * (layerCount - 1)) / 2;

      const newNeurons = [];
      const positionMap = {};

      for (let l = 0; l < layerCount; l++) {
        const n = architecture[l];
        const x = startX + l * spacing;
        const layerSpacing = this.height / (n + 1);

        for (let i = 0; i < n; i++) {
          const key = `${l}-${i}`;
          const existing = existingPositions[key];

          if (existing) {
            newNeurons.push({ ...existing });
            positionMap[key] = existing;
          } else {
            const neuron = {
              x,
              y: layerSpacing * (i + 1),
              layer: l,
              index: i,
              radius: NEURON_BASE_RADIUS,
              activation: 0,
              dragging: false,
              key,
            };
            newNeurons.push(neuron);
            positionMap[key] = neuron;
          }
        }
      }

      this.neurons = newNeurons;
      return positionMap;
    }

    updateActivations(activations) {
      for (const neuron of this.neurons) {
        const layerActs = activations[neuron.layer];
        if (layerActs) {
          const val = layerActs[neuron.index];
          neuron.activation = val !== undefined ? val : 0;
        }
      }
    }

    spawnSignals(architecture) {
      if (!this.showSignals) return;
      for (let l = 0; l < architecture.length - 1; l++) {
        const fromNeurons = this.neurons.filter((n) => n.layer === l);
        const toNeurons = this.neurons.filter((n) => n.layer === l + 1);
        if (fromNeurons.length === 0 || toNeurons.length === 0) continue;

        const from = fromNeurons[Math.floor(Math.random() * fromNeurons.length)];
        const to = toNeurons[Math.floor(Math.random() * toNeurons.length)];

        this.signals.push({
          fromX: from.x,
          fromY: from.y,
          toX: to.x,
          toY: to.y,
          progress: 0,
          color: from.activation > 0 ? COLORS.teal : COLORS.salmon,
        });
      }
    }

    updateSignals() {
      this.signals = this.signals.filter((s) => {
        s.progress += SIGNAL_SPEED * (1 + Math.abs(s.progress - 0.5));
        return s.progress < 1;
      });
    }

    drawBackground() {
      const ctx = this.ctx;
      ctx.fillStyle = COLORS.grid;
      const gridSize = 40;
      for (let x = 0; x < this.width; x += gridSize) {
        ctx.fillRect(x, 0, 1, this.height);
      }
      for (let y = 0; y < this.height; y += gridSize) {
        ctx.fillRect(0, y, this.width, 1);
      }
    }

    drawConnections(weights, architecture) {
      if (!this.showWeights) return;
      const ctx = this.ctx;

      for (let l = 0; l < architecture.length - 1; l++) {
        const fromNeurons = this.neurons.filter((n) => n.layer === l);
        const toNeurons = this.neurons.filter((n) => n.layer === l + 1);
        const w = weights[l];
        if (!w) continue;

        const kernel = w.kernel;
        let maxW = 1;
        for (const row of kernel) {
          for (const val of row) {
            maxW = Math.max(maxW, Math.abs(val));
          }
        }

        for (let i = 0; i < fromNeurons.length; i++) {
          for (let j = 0; j < toNeurons.length; j++) {
            const from = fromNeurons[i];
            const to = toNeurons[j];
            const weight = kernel[i] ? kernel[i][j] || 0 : 0;
            const normalized = weight / maxW;
            const thickness = Math.abs(normalized) * 3 + 0.5;

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);

            if (normalized > 0) {
              ctx.strokeStyle = `rgba(111, 157, 154, ${Math.abs(normalized) * 0.6 + 0.08})`;
            } else {
              ctx.strokeStyle = `rgba(217, 139, 115, ${Math.abs(normalized) * 0.6 + 0.08})`;
            }
            ctx.lineWidth = thickness;
            ctx.stroke();
          }
        }
      }
    }

    drawSignals() {
      const ctx = this.ctx;
      for (const s of this.signals) {
        const x = s.fromX + (s.toX - s.fromX) * s.progress;
        const y = s.fromY + (s.toY - s.fromY) * s.progress;
        const alpha = 1 - Math.abs(s.progress - 0.5) * 2;

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
        gradient.addColorStop(0, s.color);
        gradient.addColorStop(1, "transparent");

        ctx.beginPath();
        ctx.arc(x, y, 8 * alpha, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }

    drawNeurons(architecture) {
      const ctx = this.ctx;

      for (const neuron of this.neurons) {
        const { x, y, activation, radius, layer, index } = neuron;
        const isInput = layer === 0;
        const isOutput = layer === architecture.length - 1;

        const glowSize = radius * 2.5;
        const glowGradient = ctx.createRadialGradient(x, y, radius * 0.5, x, y, glowSize);

        if (activation > 0) {
          glowGradient.addColorStop(0, `rgba(111, 157, 154, ${Math.abs(activation) * 0.3})`);
        } else {
          glowGradient.addColorStop(0, `rgba(217, 139, 115, ${Math.abs(activation) * 0.3})`);
        }
        glowGradient.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(x, y, glowSize, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.neuronFill;
        ctx.fill();

        const strokeAlpha = 0.4 + Math.abs(activation) * 0.6;
        if (isInput) {
          ctx.strokeStyle = `rgba(214, 179, 71, ${strokeAlpha})`;
        } else if (isOutput) {
          ctx.strokeStyle = `rgba(217, 139, 115, ${strokeAlpha})`;
        } else {
          ctx.strokeStyle = `rgba(111, 157, 154, ${strokeAlpha})`;
        }
        ctx.lineWidth = 2;
        ctx.stroke();

        if (neuron.dragging) {
          ctx.beginPath();
          ctx.arc(x, y, radius + 5, 0, Math.PI * 2);
          ctx.strokeStyle = COLORS.mustard;
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        ctx.fillStyle = COLORS.text;
        ctx.font = `${isInput || isOutput ? "bold 11" : "10"}px Fraunces, Inter, system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        if (isInput) {
          ctx.fillText(`x${index + 1}`, x, y);
        } else if (isOutput) {
          ctx.fillText("out", x, y);
        } else {
          const actStr = activation >= 0 ? "+" : "";
          ctx.fillText(`${actStr}${activation.toFixed(2)}`, x, y);
        }

        if (isInput || isOutput) {
          ctx.fillStyle = COLORS.textMuted;
          ctx.font = "9px Fraunces, Inter, system-ui, sans-serif";
          const label = isInput ? `Input ${index + 1}` : "Output";
          ctx.fillText(label, x, y - radius - 10);
        }
      }
    }

    drawLayerLabels(architecture) {
      const ctx = this.ctx;
      const layerLabels = ["Input", ...architecture.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), "Output"];

      const layerXPositions = {};
      for (const n of this.neurons) {
        if (!layerXPositions[n.layer]) layerXPositions[n.layer] = [];
        layerXPositions[n.layer].push(n.x);
      }

      for (let l = 0; l < architecture.length; l++) {
        const positions = layerXPositions[l];
        if (!positions || positions.length === 0) continue;
        const x = positions[0];
        const maxY = Math.max(...this.neurons.filter((n) => n.layer === l).map((n) => n.y));

        ctx.fillStyle = COLORS.textMuted;
        ctx.font = "bold 10px Fraunces, Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(layerLabels[l] || `Layer ${l}`, x, maxY + 40);
      }
    }

    drawBoundary(network, architecture) {
      if (!this.showBoundary) {
        this.boundaryCtx.clearRect(0, 0, this.width, this.height);
        return;
      }

      const res = 60;
      const cellW = this.width / res;
      const cellH = this.height / res;

      const inputNeurons = this.neurons.filter((n) => n.layer === 0);
      const outputNeurons = this.neurons.filter((n) => n.layer === architecture.length - 1);

      if (inputNeurons.length < 2 || outputNeurons.length < 1) return;

      const minX = Math.min(...inputNeurons.map((n) => n.x));
      const maxX = Math.max(...inputNeurons.map((n) => n.x));
      const minY = Math.min(...inputNeurons.map((n) => n.y));
      const maxY = Math.max(...inputNeurons.map((n) => n.y));

      const boundaryCtx = this.boundaryCtx;
      boundaryCtx.clearRect(0, 0, this.width, this.height);

      for (let gx = 0; gx < res; gx++) {
        for (let gy = 0; gy < res; gy++) {
          const nx = (gx / res) * 2 - 1;
          const ny = (gy / res) * 2 - 1;

          try {
            const pred = network.predict([[nx, ny]]);
            const val = pred[0][0];

            const r = Math.round(217 * (1 - val) + 111 * val);
            const g = Math.round(139 * (1 - val) + 157 * val);
            const b = Math.round(115 * (1 - val) + 154 * val);
            boundaryCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.18)`;
          } catch {
            boundaryCtx.fillStyle = "transparent";
          }
          boundaryCtx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    render(network, architecture) {
      this.time += 0.016;
      this.ctx.clearRect(0, 0, this.width, this.height);

      this.drawBackground();
      this.drawConnections(network.weights, architecture);
      this.updateSignals();
      this.drawSignals();
      this.drawNeurons(architecture);
      this.drawLayerLabels(architecture);
    }

    getNeuronAt(mx, my) {
      for (let i = this.neurons.length - 1; i >= 0; i--) {
        const n = this.neurons[i];
        const dx = mx - n.x;
        const dy = my - n.y;
        if (dx * dx + dy * dy <= (n.radius + 5) * (n.radius + 5)) {
          return n;
        }
      }
      return null;
    }
  }

  class App {
    constructor() {
      this.network = new NeuralNetwork();
      this.canvas = document.getElementById("network-canvas");
      this.boundaryCanvas = document.getElementById("boundary-canvas");
      this.renderer = new Renderer(this.canvas, this.boundaryCanvas);
      this.tooltip = document.getElementById("tooltip");

      this.isTraining = false;
      this.trainInterval = null;
      this.problemData = null;
      this.dragNeuron = null;
      this.dragOffset = { x: 0, y: 0 };
      this.hasDragged = false;

      this.bindControls();
      this.bindCanvas();
      this.loadData("xor");
      this.layoutAndRender();
      this.startAnimationLoop();

      window.addEventListener("resize", () => {
        this.renderer.resize();
        this.renderer.layoutNetwork(this.network.architecture, {});
        this.layoutAndRender();
      });
    }

    bindControls() {
      document.getElementById("btn-add-layer").addEventListener("click", () => {
        this.network.addLayer(4);
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("btn-remove-layer").addEventListener("click", () => {
        this.network.removeLayer();
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("btn-add-neuron").addEventListener("click", () => {
        this.network.addNeuron();
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("btn-remove-neuron").addEventListener("click", () => {
        this.network.removeNeuron();
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("activation-select").addEventListener("change", (e) => {
        this.network.activation = e.target.value;
        this.network.buildModel();
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("problem-select").addEventListener("change", (e) => {
        this.loadData(e.target.value);
        this.layoutAndRender();
      });

      const lrSlider = document.getElementById("lr-slider");
      const lrValue = document.getElementById("lr-value");
      lrSlider.addEventListener("input", () => {
        const lr = Math.pow(10, parseFloat(lrSlider.value));
        lrValue.textContent = lr.toFixed(4);
        this.network.model.compile({
          optimizer: tf.train.adam(lr),
          loss: "binaryCrossentropy",
        });
      });

      const speedSlider = document.getElementById("speed-slider");
      const speedValue = document.getElementById("speed-value");
      speedSlider.addEventListener("input", () => {
        speedValue.textContent = speedSlider.value;
      });

      document.getElementById("btn-train").addEventListener("click", () => {
        this.toggleTraining();
      });

      document.getElementById("btn-reset").addEventListener("click", () => {
        this.stopTraining();
        this.network.reset();
        this.layoutAndRender();
        this.updateInfo();
      });

      document.getElementById("show-weights").addEventListener("change", (e) => {
        this.renderer.showWeights = e.target.checked;
      });

      document.getElementById("show-signals").addEventListener("change", (e) => {
        this.renderer.showSignals = e.target.checked;
      });

      document.getElementById("show-boundary").addEventListener("change", (e) => {
        this.renderer.showBoundary = e.target.checked;
        if (!e.target.checked) {
          this.renderer.boundaryCtx.clearRect(0, 0, this.renderer.width, this.renderer.height);
        }
      });
    }

    bindCanvas() {
      const canvas = this.canvas;

      canvas.addEventListener("mousedown", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const neuron = this.renderer.getNeuronAt(mx, my);
        this.hasDragged = false;

        if (neuron) {
          this.dragNeuron = neuron;
          neuron.dragging = true;
          this.dragOffset.x = mx - neuron.x;
          this.dragOffset.y = my - neuron.y;
          canvas.classList.add("dragging");
        }
      });

      canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (this.dragNeuron) {
          this.hasDragged = true;
          this.dragNeuron.x = mx - this.dragOffset.x;
          this.dragNeuron.y = my - this.dragOffset.y;
          this.dragNeuron.x = Math.max(this.dragNeuron.radius, Math.min(this.renderer.width - this.dragNeuron.radius, this.dragNeuron.x));
          this.dragNeuron.y = Math.max(this.dragNeuron.radius, Math.min(this.renderer.height - this.dragNeuron.radius, this.dragNeuron.y));
        } else {
          const neuron = this.renderer.getNeuronAt(mx, my);
          if (neuron) {
            this.tooltip.style.display = "block";
            this.tooltip.style.left = mx + 15 + "px";
            this.tooltip.style.top = my - 10 + "px";
            const actStr = neuron.activation >= 0 ? "+" : "";
            this.tooltip.textContent = `L${neuron.layer}·N${neuron.index} | activation: ${actStr}${neuron.activation.toFixed(4)}`;
          } else {
            this.tooltip.style.display = "none";
          }
        }
      });

      canvas.addEventListener("mouseup", () => {
        if (this.dragNeuron) {
          this.dragNeuron.dragging = false;
          this.dragNeuron = null;
          canvas.classList.remove("dragging");
        }
      });

      canvas.addEventListener("mouseleave", () => {
        if (this.dragNeuron) {
          this.dragNeuron.dragging = false;
          this.dragNeuron = null;
          canvas.classList.remove("dragging");
        }
        this.tooltip.style.display = "none";
      });

      canvas.addEventListener("click", (e) => {
        if (this.hasDragged) return;
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const neuron = this.renderer.getNeuronAt(mx, my);

        if (neuron && neuron.layer === 0) {
          this.pokeInputNeuron(neuron);
        }
      });

      canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mx = touch.clientX - rect.left;
        const my = touch.clientY - rect.top;
        const neuron = this.renderer.getNeuronAt(mx, my);

        if (neuron) {
          this.dragNeuron = neuron;
          neuron.dragging = true;
          this.dragOffset.x = mx - neuron.x;
          this.dragOffset.y = my - neuron.y;
        } else {
          const inputNeuron = this.renderer.neurons.find(
            (n) => n.layer === 0 && this.renderer.getNeuronAt(mx, my) === n
          );
          if (!inputNeuron) {
            const closest = this.renderer.neurons.filter((n) => n.layer === 0).sort((a, b) => {
              const da = (mx - a.x) ** 2 + (my - a.y) ** 2;
              const db = (mx - b.x) ** 2 + (my - b.y) ** 2;
              return da - db;
            })[0];
            if (closest) this.pokeInputNeuron(closest);
          }
        }
      }, { passive: false });

      canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        if (!this.dragNeuron) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mx = touch.clientX - rect.left;
        const my = touch.clientY - rect.top;

        this.dragNeuron.x = mx - this.dragOffset.x;
        this.dragNeuron.y = my - this.dragOffset.y;
        this.dragNeuron.x = Math.max(this.dragNeuron.radius, Math.min(this.renderer.width - this.dragNeuron.radius, this.dragNeuron.x));
        this.dragNeuron.y = Math.max(this.dragNeuron.radius, Math.min(this.renderer.height - this.dragNeuron.radius, this.dragNeuron.y));
      }, { passive: false });

      canvas.addEventListener("touchend", () => {
        if (this.dragNeuron) {
          this.dragNeuron.dragging = false;
          this.dragNeuron = null;
        }
      });
    }

    pokeInputNeuron(neuron) {
      const input = [0.5, 0.5];
      if (neuron.index < input.length) {
        input[neuron.index] = input[neuron.index] > 0.5 ? 0 : 1;
      }
      this.renderer.sampleInput = input;

      try {
        const activations = this.network.getActivations(input);
        this.renderer.updateActivations(activations);
        for (let i = 0; i < 5; i++) {
          this.renderer.spawnSignals(this.network.architecture);
        }
      } catch {
        // ignore
      }
    }

    loadData(problem) {
      switch (problem) {
        case "xor":
          this.problemData = DataGenerator.xor(200);
          break;
        case "circle":
          this.problemData = DataGenerator.circle(200);
          break;
        case "spiral":
          this.problemData = DataGenerator.spiral(300);
          break;
      }
    }

    async toggleTraining() {
      if (this.isTraining) {
        this.stopTraining();
      } else {
        this.startTraining();
      }
    }

    async startTraining() {
      this.isTraining = true;
      const btn = document.getElementById("btn-train");
      btn.textContent = "Stop";
      btn.classList.add("active");
      document.getElementById("stat-status").textContent = "Training";
      document.getElementById("stat-status").style.color = "#9fb06f";

      const speed = parseInt(document.getElementById("speed-slider").value);

      this.trainDataT = tf.tensor2d(this.problemData.data);
      this.trainLabelsT = tf.tensor2d(this.problemData.labels, [this.problemData.labels.length, 1]);

      let boundaryCounter = 0;

      const trainLoop = async () => {
        if (!this.isTraining) return;

        for (let i = 0; i < speed; i++) {
          const loss = await this.network.trainStep(this.trainDataT, this.trainLabelsT);
          this.network.epoch++;
          this.network.loss = loss;
        }

        this.updateInfo();
        this.updateActivations();
        this.renderer.spawnSignals(this.network.architecture);

        boundaryCounter++;
        if (this.renderer.showBoundary && boundaryCounter % 10 === 0) {
          this.renderer.drawBoundary(this.network, this.network.architecture);
        }

        if (this.isTraining) {
          requestAnimationFrame(trainLoop);
        }
      };

      requestAnimationFrame(trainLoop);
    }

    stopTraining() {
      this.isTraining = false;
      if (this.trainDataT) { this.trainDataT.dispose(); this.trainDataT = null; }
      if (this.trainLabelsT) { this.trainLabelsT.dispose(); this.trainLabelsT = null; }
      const btn = document.getElementById("btn-train");
      btn.textContent = "Train";
      btn.classList.remove("active");
      document.getElementById("stat-status").textContent = "Idle";
      document.getElementById("stat-status").style.color = "";
    }

    updateActivations() {
      try {
        const activations = this.network.getActivations(this.renderer.sampleInput);
        this.renderer.updateActivations(activations);
      } catch {
        // ignore
      }
    }

    layoutAndRender() {
      const existingPositions = {};
      for (const n of this.renderer.neurons) {
        existingPositions[n.key] = n;
      }
      this.renderer.layoutNetwork(this.network.architecture, existingPositions);
      this.updateActivations();
      this.updateInfo();
    }

    updateInfo() {
      document.getElementById("info-layers").textContent = this.network.architecture.length;
      document.getElementById("info-params").textContent = this.network.getParamCount().toLocaleString();
      document.getElementById("info-epoch").textContent = this.network.epoch.toLocaleString();
      document.getElementById("info-loss").textContent =
        this.network.loss !== null ? this.network.loss.toFixed(6) : "\u2014";

      document.getElementById("stat-epoch").textContent = this.network.epoch.toLocaleString();
      document.getElementById("stat-loss").textContent =
        this.network.loss !== null ? this.network.loss.toFixed(6) : "\u2014";
      document.getElementById("stat-params").textContent = this.network.getParamCount().toLocaleString();
    }

    startAnimationLoop() {
      const loop = () => {
        this.renderer.render(this.network, this.network.architecture);
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }

  new App();
})();
