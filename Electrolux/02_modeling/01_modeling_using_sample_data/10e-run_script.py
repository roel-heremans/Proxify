# Databricks notebook source
# MAGIC %md
# MAGIC ## The effect of the sample size (i.e. the average qty_sold) on the regression performance (i.e. MAPE)
# MAGIC The different product hierarchies (line-group-subgroup) are show in different colors. Hovering over the datapoints shows you the product name under consideration.
# MAGIC One can clearly see that the more qty_sold volume you have the better the SCR prediction becomes reflecting into smaller MAPE. The four product lines are at the upper-left
# MAGIC corner (high sample size - low mape [good prediction performanc] )

# COMMAND ----------

from utils.util import make_sample_size_study
make_sample_size_study()

# COMMAND ----------


