from django import forms

class UploadCSVForm(forms.Form):
    # 1. Le champ de fichier
    # Correspond à <input type="file" name="csv_file" ... required>
    csv_file = forms.FileField(label='Sélectionnez le fichier CSV')
    
    # 2. En-tête (Header)
    # Correspond à <input type="radio" name="has_header" ... required>
    # C'est une chaîne ('yes' ou 'no')
    has_header = forms.CharField(label='Contient un en-tête')
    
    # 3. Délimiteur
    # Correspond à <input type="text" name="delimiter" ... required>
    delimiter = forms.CharField(label='Caractère séparateur', max_length=5)
    
    # 4. Méthode d'imputation
    # Correspond à <input type="radio" name="imputation_method" ... required>
    # C'est une chaîne ('intelligente', 'moyenne' ou 'mediane')
    imputation_method = forms.CharField(label='Méthode d\'imputation')
    
    # 5. Standardisation
    # Correspond à <input type="radio" name="standardize" ... required>
    # C'est une chaîne ('yes' ou 'no')
    standardize = forms.CharField(label='Standardisation')
    
    # 6. Noms des colonnes
    # Correspond à <input type="text" name="column_names" ...>
    # Bien qu'il soit conditionnellement requis en JavaScript, nous le définissons
    # comme NON requis (required=False) dans Django, car il est absent si l'en-tête existe.
    column_names = forms.CharField(label='Noms des colonnes', required=False)