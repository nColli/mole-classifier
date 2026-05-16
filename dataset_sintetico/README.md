# Dataset Sintético

Debido a que el modelo entrendado con imagenes recortadas tiene bajo rendimiento en imagenes reales donde aparece piel, partes del cuerpo y otros elementos,
se crea un dataset sintetico en el que se inserta las imagenes recortadas de lunares en imagenes de piel y partes del cuerpo.
El dataset utilizado tambien posee imagenes de otras enfermedades de la piel por lo que puede ser util probar entrenar el modelo con estas imagenes para evitar 
que se detecten como lunares y bajar los falsos positivos.

## Datasets
[Dataset skin](https://app.roboflow.com/nicolass-workspace-jvqig/skin-1qn4y-iaecv)
[Dataset mole](https://app.roboflow.com/nicolass-workspace-jvqig/mole-classification-3zt76)
