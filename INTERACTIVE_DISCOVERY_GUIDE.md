# Interactive Discovery Setup Guide

## 🚀 Quick Setup

### Step 1: Create the Script Files

Save the provided artifacts as these files:

1. **Interactive Discovery Engine** → `scripts/interactive_discovery.py`
2. **Discovery CLI** → `scripts/discover_patterns.py`  
3. **Enhanced Examples** → `examples/enhanced_discovery_examples.py`

### Step 2: Test the Setup

```bash
# Test basic functionality
python scripts/discover_patterns.py list

# Test interactive mode
python scripts/interactive_discovery.py
```

### Step 3: Generate Some Test Data (if none exists)

```bash
# Generate test datasets
python src/generators/universal_generator.py interactive

# Or create a quick demo dataset
python src/generators/universal_generator.py demo
```

### Step 4: Start Discovering Patterns!

```bash
# Full interactive mode (recommended)
python scripts/interactive_discovery.py

# Command-line mode
python scripts/discover_patterns.py interactive
```

## 🎯 Usage Examples

### Interactive Mode
```bash
python scripts/interactive_discovery.py
```

This will:
- Scan for all available datasets
- Show you what's available in a nice format
- Let you choose analysis options
- Guide you through the discovery process

### Command Line Mode
```bash
# List available datasets
python scripts/discover_patterns.py list

# Analyze specific dataset
python scripts/discover_patterns.py analyze data/output/rule_name/dataset.csv

# Analyze entire folder
python scripts/discover_patterns.py folder data/output/rule_name/

# Quick analysis
python scripts/discover_patterns.py analyze dataset.csv --quick

# Deep analysis with embeddings
python scripts/discover_patterns.py analyze dataset.csv --embedding fourier
```

## 📊 Dataset Types Supported

The interactive discovery engine automatically detects and handles:

- **Algebraic Features** - Comprehensive mathematical properties
- **Digit Tensors** - Digit-based patterns with optional embeddings  
- **Sequence Patterns** - Gap analysis and local density features
- **Prefix-Suffix Matrices** - Structural digit pattern analysis
- **Legacy ML Datasets** - Original format datasets

## 🔧 Analysis Modes

- **Quick** - Fast analysis with 10K samples
- **Standard** - Balanced approach with 50K samples  
- **Deep** - Thorough analysis with 100K samples + embeddings
- **Custom** - Full control over all parameters

## 📁 Output Files

Results are saved as:
- `interactive_discovery_results.json` - Full interactive session results
- `discovery_results_<dataset>.json` - Individual dataset analysis
- `matrix_analysis_<dataset>.json` - Matrix-specific analysis

## 🚨 Troubleshooting

### "No datasets found"
- Run the Universal Dataset Generator first: `python src/generators/universal_generator.py interactive`
- Check that datasets exist in `data/output/` or `data/raw/`

### Import errors
- Make sure you're running from the project root directory
- Install dependencies: `pip install -e .`

### Performance issues
- Use quick mode: `--quick`
- Reduce sample size: `--max-samples 10000`
- Analyze smaller datasets first

## 💡 Tips

1. **Start Small** - Generate a small test dataset first
2. **Use Interactive Mode** - It guides you through everything
3. **Quick Mode** - Use for initial exploration
4. **Save Results** - Always save for later comparison
5. **Batch Analysis** - Use folder mode for multiple datasets

## 🎉 Next Steps

Once you have the interactive discovery working:

1. Generate datasets for your mathematical rules
2. Use the discovery engine to find patterns
3. Compare different dataset formats
4. Export results for further analysis
5. Integrate with your research workflow

Happy discovering! 🧮✨
