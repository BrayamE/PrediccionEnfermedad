import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;

public class PrediccionEnfermedades {

    public static void main(String[] args) throws Exception {
        // Cargar el conjunto de datos
        DataSource source = new DataSource("ruta/del/conjunto/de/datos.arff"); // Reemplaza con la ruta de tu conjunto de datos
        Instances data = source.getDataSet();

        // Establecer el índice del atributo objetivo (enfermedad)
        data.setClassIndex(data.numAttributes() - 1);

        // Crear y entrenar el clasificador de árbol de decisión (J48)
        Classifier tree = new J48();
        tree.buildClassifier(data);

        // Crear una instancia de prueba (reemplaza con tus valores)
        Instance inst = new DenseInstance(data.numAttributes());
        inst.setValue(data.attribute("Atributo1"), valor1); // Reemplaza con tus valores
        inst.setValue(data.attribute("Atributo2"), valor2); // Reemplaza con tus valores
        // ...

        // Realizar la predicción
        double prediction = tree.classifyInstance(inst);

        // Obtener la etiqueta de la clase predicha
        String predictedClass = data.classAttribute().value((int) prediction);

        System.out.println("La enfermedad predicha es: " + predictedClass);
    }
}
