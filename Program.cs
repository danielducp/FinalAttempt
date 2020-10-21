using System;
using System.IO;
using Microsoft.ML;
namespace FinalAttempt
{
    class Program
    {

        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "IllnessTraining.csv");
static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory,  "IllnessTest.csv");


        static void Main(string[] args)
        {
MLContext mlContext = new MLContext(seed: 0);
var model = Train(mlContext, _trainDataPath);


        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
{
IDataView dataView = mlContext.Data.LoadFromTextFile<IllnessType>(dataPath, hasHeader: true, separatorChar: ',');

var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName:"NameOfIllness")
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "FeverEncoded", inputColumnName:"Fever"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "FatigueEncoded", inputColumnName: "Fatigue"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CoughEncoded", inputColumnName: "Cough"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LossOfSensesEncoded", inputColumnName: "LossOfSenses"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SneezingEncoded", inputColumnName: "Sneezing"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AchesandPainsEncoded", inputColumnName: "AchesandPains"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RunnyOrStuffyNoseEncoded", inputColumnName: "RunnyOrStuffyNose"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SoreThroatEncoded", inputColumnName: "SoreThroat"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DiarrhoeaEncoded", inputColumnName: "Diarrhoea"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HeadachesEncoded", inputColumnName: "Headaches"))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ShortnessOfBreathEncoded", inputColumnName: "ShortnessOfBreath"))
.Append(mlContext.Transforms.Concatenate("Features", "FeverEncoded", "FatigueEncoded", "CoughEncoded", "LossOfSensesEncoded", "SneezingEncoded",  "AchesandPainsEncoded", "RunnyOrStuffyNoseEncoded", "SoreThroatEncoded", "DiarrhoeaEncoded", "HeadachesEncoded", "ShortnessOfBreathEncoded"))

.Append(mlContext.Regression.Trainers.FastTree());

var model = pipeline.Fit(dataView);
return model;
}

private static void Evaluate(MLContext mlContext, ITransformer model)
{
Evaluate(mlContext, model);
IDataView dataView = mlContext.Data.LoadFromTextFile<IllnessType>(_testDataPath, hasHeader: true, separatorChar: ',');
var predictions = model.Transform(dataView);

var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
Console.WriteLine();
Console.WriteLine($"*************************************************");
Console.WriteLine($"*       Model quality metrics evaluation         ");
Console.WriteLine($"*------------------------------------------------");

Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
TestSinglePrediction(mlContext, model);
var predictionFunction = mlContext.Model.CreatePredictionEngine<IllnessType, IllnessPrediction>(model);

var illnessSample = new IllnessType()
{
    Fever = "Common",
    Fatigue = "Sometimes",
    Cough = "Common",
    LossOfSenses = "Common",
    Sneezing = "No",
    AchesandPains = "Sometimes",
   RunnyOrStuffyNose = "Rare",
    SoreThroat = "Sometimes",
    Diarrhoea = "Rare",
    Headaches = "Sometimes",
    ShortnessOfBreath = "Sometimes",
    NameOfIllness = "" // actual fare for this trip = 15.5
};
var prediction = predictionFunction.Predict(illnessSample);
Console.WriteLine($"**********************************************************************");
Console.WriteLine($"Predicted illness: {prediction.NameOfIllness}, actual illness: Coronavirus");
Console.WriteLine($"**********************************************************************");
}
    }
}
