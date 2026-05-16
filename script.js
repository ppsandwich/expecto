class Calculator {
  constructor(previousOperandEl, currentOperandEl) {
    this.previousOperandEl = previousOperandEl;
    this.currentOperandEl = currentOperandEl;
    this.clear();
  }

  clear() {
    this.currentOperand = '0';
    this.previousOperand = '';
    this.operator = null;
    this.shouldResetScreen = false;
  }

  delete() {
    if (this.currentOperand.length === 1 || (this.currentOperand.length === 2 && this.currentOperand.startsWith('-'))) {
      this.currentOperand = '0';
    } else {
      this.currentOperand = this.currentOperand.slice(0, -1);
    }
  }

  appendNumber(number) {
    if (this.shouldResetScreen) {
      this.currentOperand = '';
      this.shouldResetScreen = false;
    }
    if (number === '0' && this.currentOperand === '0') return;
    if (number === '.' && this.currentOperand.includes('.')) return;
    if (this.currentOperand === '0' && number !== '.') {
      this.currentOperand = number;
    } else {
      this.currentOperand += number;
    }
  }

  chooseOperator(operator) {
    if (this.currentOperand === '' && this.previousOperand === '') return;
    if (this.previousOperand !== '' && !this.shouldResetScreen) {
      this.compute();
    }
    this.operator = operator;
    this.previousOperand = this.currentOperand;
    this.currentOperand = '';
    this.shouldResetScreen = false;
  }

  compute() {
    const prev = parseFloat(this.previousOperand);
    const current = parseFloat(this.currentOperand);
    if (isNaN(prev) || isNaN(current)) return;

    let result;
    switch (this.operator) {
      case '+':
        result = prev + current;
        break;
      case '-':
        result = prev - current;
        break;
      case '*':
        result = prev * current;
        break;
      case '/':
        if (current === 0) {
          this.currentOperand = 'Error';
          this.previousOperand = '';
          this.operator = null;
          this.shouldResetScreen = true;
          return;
        }
        result = prev / current;
        break;
      default:
        return;
    }

    this.currentOperand = this.formatResult(result);
    this.previousOperand = '';
    this.operator = null;
    this.shouldResetScreen = true;
  }

  formatResult(number) {
    if (Number.isInteger(number) && Math.abs(number) < 1e15) {
      return number.toString();
    }
    const str = number.toPrecision(10);
    return parseFloat(str).toString();
  }

  updateDisplay() {
    this.currentOperandEl.textContent = this.getDisplayNumber(this.currentOperand);
    if (this.operator) {
      const opSymbol = { '/': '÷', '*': '×', '-': '−', '+': '+' }[this.operator];
      this.previousOperandEl.textContent = `${this.getDisplayNumber(this.previousOperand)} ${opSymbol}`;
    } else {
      this.previousOperandEl.textContent = this.previousOperand
        ? this.getDisplayNumber(this.previousOperand)
        : '';
    }
  }

  getDisplayNumber(number) {
    if (number === 'Error') return 'Error';
    if (number === '' || number === undefined) return '';
    const stringNumber = number.toString();
    if (stringNumber.endsWith('.')) return this.addCommas(stringNumber.slice(0, -1)) + '.';
    if (stringNumber.includes('.')) {
      const [integer, decimal] = stringNumber.split('.');
      return this.addCommas(integer) + '.' + decimal;
    }
    return this.addCommas(stringNumber);
  }

  addCommas(integerString) {
    if (integerString === '' || integerString === '-') return integerString;
    const isNegative = integerString.startsWith('-');
    const digits = isNegative ? integerString.slice(1) : integerString;
    const formatted = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return isNegative ? '-' + formatted : formatted;
  }
}

const previousOperandEl = document.getElementById('previousOperand');
const currentOperandEl = document.getElementById('currentOperand');
const calculator = new Calculator(previousOperandEl, currentOperandEl);

document.querySelectorAll('.btn').forEach(button => {
  button.addEventListener('click', () => {
    const action = button.dataset.action;
    const value = button.dataset.value;

    switch (action) {
      case 'number':
        calculator.appendNumber(value);
        break;
      case 'operator':
        calculator.chooseOperator(value);
        break;
      case 'equals':
        calculator.compute();
        break;
      case 'clear':
        calculator.clear();
        break;
      case 'delete':
        calculator.delete();
        break;
      case 'decimal':
        calculator.appendNumber('.');
        break;
    }
    calculator.updateDisplay();
  });
});

document.addEventListener('keydown', e => {
  if (e.key >= '0' && e.key <= '9') {
    calculator.appendNumber(e.key);
  } else if (e.key === '.') {
    calculator.appendNumber('.');
  } else if (e.key === '+' || e.key === '-' || e.key === '*' || e.key === '/') {
    calculator.chooseOperator(e.key);
  } else if (e.key === 'Enter' || e.key === '=') {
    e.preventDefault();
    calculator.compute();
  } else if (e.key === 'Backspace') {
    calculator.delete();
  } else if (e.key === 'Escape') {
    calculator.clear();
  } else {
    return;
  }
  calculator.updateDisplay();
});
