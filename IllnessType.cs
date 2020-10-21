using Microsoft.ML.Data;

namespace FinalAttempt
{
    public class IllnessType{
    [LoadColumn(0)]    
        public string Fever;    
    
        [LoadColumn(1)]    
        public string Fatigue;    
    
        [LoadColumn(2)]    
        public string Cough;    
    
        [LoadColumn(3)]    
        public string LossOfSenses;    
    
        [LoadColumn(4)]    
        public string Sneezing;    
    
        [LoadColumn(5)]    
        public string AchesandPains; 

                [LoadColumn(6)]    
        public string RunnyOrStuffyNose;  

                [LoadColumn(7)]    
        public string SoreThroat;  

        
                [LoadColumn(8)]    
        public string Diarrhoea;   

                
                [LoadColumn(9)]    
        public string Headaches;  

                
                [LoadColumn(10)]    
        public string ShortnessOfBreath;  
                        [LoadColumn(11)]    
        public string NameOfIllness; 
}

public class IllnessPrediction{
      [ColumnName("IllnessName")]    
      public string NameOfIllness; 


}


}