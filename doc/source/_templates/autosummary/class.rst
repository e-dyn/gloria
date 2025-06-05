{{ fullname | escape | underline}}

{% set excluded_attributes = ['model_config', 'model_fields', 'model_extra', 'model_fields_set', 'model_computed_fields'] %}
{% set excluded_members = [
    'Foster',
    'make_all_features',
    'model_config',
    '__init__',
    'preprocess',
    'set_changepoints',
    'time_to_integer',
    'validate_changepoints',
    'validate_column_name',
    'validate_dataframe',
    'validate_metric_column',
    'validate_population_name',
    'validate_sampling_period'
] %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes%}
      {% if item in members 
         and item not in excluded_members 
         and item not in inherited_members %}
        ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods%}      
      {% if item in members 
         and item not in excluded_members 
         and item not in inherited_members %}
        ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   
   
   