# Project Name

This project involves using Plotly for data visualization in Jupyter Notebooks.

## Running `detailed_workflow.ipynb`

When running the `detailed_workflow.ipynb` notebook, you might encounter the following error:

### Error Details

```plaintext
ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
```

#### Full Error Traceback

```plaintext
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[7], line 14
      1 import plotly.graph_objects as go
      3 fig = go.Figure(
      4     data=[
      5         go.Candlestick(
   (...)
     12     ]
     13 )
---> 14 fig.show()

File ~/miniconda3/envs/qlib/lib/python3.8/site-packages/plotly/basedatatypes.py:3410, in BaseFigure.show(self, *args, **kwargs)
   3377 \"\"\"
   3378 Show a figure using either the default renderer(s) or the renderer(s)
   3379 specified by the renderer argument
   (...)
   3406 None
   3407 \"\"\"
   3408 import plotly.io as pio
-> 3410 return pio.show(self, *args, **kwargs)

File ~/miniconda3/envs/qlib/lib/python3.8/site-packages/plotly/io/_renderers.py:394, in show(fig, renderer, validate, **kwargs)
    389         raise ValueError(
    390             \"Mime type rendering requires ipython but it is not installed\"
    391         )
    393     if not nbformat or Version(nbformat.__version__) < Version(\"4.2.0\"):
--> 394         raise ValueError(
    395             \"Mime type rendering requires nbformat>=4.2.0 but it is not installed\"
    396         )
    398     ipython_display.display(bundle, raw=True)
    400 # external renderers

ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
```

### Solution

To resolve this error, you need to install the `nbformat` package with a version greater than or equal to 4.2.0. Follow these steps:

1. Open a terminal or command prompt.
2. Run the following command to install the correct version of `nbformat`:

    ```bash
    pip install nbformat==4.2.0
    ```

3. Restart the Jupyter kernel to apply the changes.

### Restarting the Kernel

After installing the package, ensure you restart the kernel to apply the changes. You can do this from the Jupyter Notebook interface:

1. Click on the `Kernel` menu.
2. Select `Restart` to restart the kernel.

After restarting the kernel, you should be able to run the notebook without encountering the error.

### Full Error Traceback

```plaintext
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mlflow 1.30.0 requires pandas<2, but you have pandas 2.0.3 which is incompatible.
```
### Solution
```bash
pip install pandas==1.5.3
pip install yahooquery==2.3.2
```