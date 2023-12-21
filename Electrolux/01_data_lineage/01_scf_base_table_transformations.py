# Databricks notebook source
# MAGIC %md
# MAGIC ###team_group_quality.product_max_ingested_batch table
# MAGIC ######getting maximun ingested_batch to get the lastest data

# COMMAND ----------

from pyspark.sql.functions import col, concat, lit


df = spark.sql("select product as pnc_sku, max(ingested_batch) as max_ingested_batch from qes.product_properties_base group by pnc_sku")

df2 = df.withColumn("product_record_id",concat(col("pnc_sku"),lit("-"), col("max_ingested_batch")))

df2.write.format("delta").mode("overwrite").option("overwriteSchema","true").option("path", "abfss://teamspace@elxa2ns0017.dfs.core.windows.net/group_quality/databricks_tables/data_science/product_max_ingested_batch").saveAsTable("team_group_quality.product_max_ingested_batch")


# COMMAND ----------

# MAGIC %md
# MAGIC ##team_group_quality.product_properties table

# COMMAND ----------

df_all = spark.sql("select product as pnc_sku, ingested_batch, gph_prodlinename, gph_prodgrpname, gph_prodsubgrpname from qes.product_properties_base")

df2_all = df_all.withColumn("product_record_id",concat(col("pnc_sku"),lit("-"), col("ingested_batch")))

df2_all.write.format("delta").mode("overwrite").option("overwriteSchema","true").option("path", "abfss://teamspace@elxa2ns0017.dfs.core.windows.net/group_quality/databricks_tables/data_science/product_properties").saveAsTable("team_group_quality.product_properties")

# COMMAND ----------

# MAGIC %md
# MAGIC ##team_group_quality.product_joined_dim table

# COMMAND ----------

joindf = df2.join(df2_all,df2.product_record_id == df2_all.product_record_id,"left").drop(df2_all.product_record_id).drop(df2_all.pnc_sku).drop(df2_all.ingested_batch)

print(f"Row count df2 = {df2.distinct().count()}")
print(f"Row count after join joindf = {joindf.distinct().count()}")

joindf.write.format("delta").mode("overwrite").option("overwriteSchema","true").option("path", "abfss://teamspace@elxa2ns0017.dfs.core.windows.net/group_quality/databricks_tables/data_science/product_joined_dim").saveAsTable("team_group_quality.product_joined_dim")

# COMMAND ----------

# MAGIC %md
# MAGIC ##team_group_quality.service_call_forecasting table

# COMMAND ----------

from pyspark.sql.functions import col, concat, lit
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

base_df = spark.sql("""select a.plant_key,
       a.period,
      to_date(concat(cast(a.period as string),'01'), 'yyyyMMdd') as period_as_date,
       c.market_name as market,
       a.company_code,
       d.description as warranty_category, 
       e.description as service_type, 
       a.pnc_sku,
       a.ser_no1,
       a.ser_no2,
       a.instal_per as sell_out_date, 
       to_date(a.order_date, 'yyyyMMdd') as failure_date,
       a.ttf_inst as ttfi, 
       cast(a.qty_serv as int), 
       b.gph_prodlinename as product_line, 
       b.gph_prodgrpname as product_group, 
       b.gph_prodsubgrpname as product_subgroup,
       a.order_nr,
       f.pnc_sku as pnc_sku_sales,
       f.period_dt,
       f.qty_sold,
       a.warranty_key
from qes.service_tickets_base a
left join team_group_quality.product_joined_dim b on a.pnc_sku = b.pnc_sku
left join qes.market_dim_base c on a.company_code = c.company_code
left join qes.order_type_dim_base d on a.warranty_key = d.order_type
left join qes.service_type_dim_base e on a.service_key = e.action
right join qes.aggregated_sales_base f on a.pnc_sku = f.pnc_sku
where a.period >= '201501'
and a.ttf_inst != -1 """)

#replacing nulls in plant_key, ser_no1 and ser_no2 to "none"
base_df2 = base_df.fillna("none", subset=['ser_no1']).fillna("none", subset = ['ser_no2']).fillna("none", subset = ['plant_key']).fillna("none", subset = ['sell_out_date']).fillna("none", subset = ['failure_date']).fillna("none", subset = ['order_nr'])

base_df3 = base_df2.withColumn("product_unit_key",concat(col("plant_key"),lit("-"), col("pnc_sku"),lit("-"),col("ser_no1"),lit("-"),col("ser_no2")))

#dropping plant_key, ser_no1 and ser_no2
base_df4 = base_df3.drop(*('plant_key','ser_no1','ser_no2'))

#base_df2.schema

base_df4.createOrReplaceTempView("base_df2_view")

#removing rows having company_code = 'AUW' OR 'ARP' and pnc_sku = 'ffec3025lb' or 'Poli-FDH-3497' or 'titan' or 'super'
base_df5 = spark.sql("select * from base_df2_view where company_code not in ('AUW','ARP') and pnc_sku not in ('ffec3025lb', 'Poli-FDH-3497', 'titan', 'super') ")

base_df5.write.format("delta").mode("overwrite").option("overwriteSchema","true").option("path","abfss://teamspace@elxa2ns0017.dfs.core.windows.net/group_quality/databricks_tables/data_science/service_call_forecasting").saveAsTable("team_group_quality.service_call_forecasting")


# COMMAND ----------


