from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),

    # partie regression logisitique
    path('regLog_details/', views.regLog_details, name='regLog_details'),
    path('regLog_atelier/', views.regLog_atelier, name='regLog_atelier'),
    path('regLog_form/', views.regLog_form, name='regLog_form'),
    path('regLog_prediction/', views.regLog_prediction, name='regLog_prediction'),
    
    # partie rendom forest classification
    path('randomFor_details/', views.randomFor_details, name='randomFor_details'),
    path('randomFor_cla_atelier/', views.randomFor_cla_atelier, name='randomFor_cla_atelier'),
    path('randomFor_cla_form/', views.randomFor_cla_form, name='randomFor_cla_form'),
    path('randomFor_cla_prediction/', views.randomFor_cla_prediction, name='randomFor_cla_prediction'),
    path('randomForest_reg_atelier/', views.randomForest_reg_atelier, name='randomForest_reg_atelier' ),
    path('randomForest_reg_form/', views.randomForest_reg_form, name='randomForest_reg_form'),
    path('randomFor_reg_prediction/', views.randomFor_reg_prediction, name='randomFor_reg_prediction'),
    
    # Partie XG-Boost regression 
    path('XGboost_details/', views.XGboost_details, name='XGboost_details'),
    path('XGboost_reg_atelier/', views.XGboost_reg_atelier, name='XGboost_reg_atelier'),
    path('XGboost_reg_form/', views.XGboost_reg_form, name='XGboost_reg_form'),
    path('XGboost_reg_prediction/', views.XGboost_reg_prediction, name='XGboost_reg_prediction'),
    path('XGboost_cla_atelier/', views.XGboost_cla_atelier, name='XGboost_cla_atelier' ),
    path('XGboost_cla_form/', views.XGboost_cla_form, name='XGboost_cla_form'),
    path('XGboost_cla_prediction/', views.XGboost_cla_prediction, name='XGboost_cla_prediction'),



    # partie regression lineaire
    path('reg_lin_details/',views.reg_lin_details, name='reg_lin_details'),    
    path('reg_lin_atelier/',views.reg_lin_atelier, name='reg_lin_atelier'),
    path('reg_lin_form/',views.reg_lin_form, name='reg_lin_form'), 
    path('reg_lin_pred/',views.reg_lin_pred, name='reg_lin_pred'),




# partie SVM
    path('SVM_details/',views.SVM_details, name='SVM_details'),

    #SVR
    path('SVR_atelier/',views.SVR_atelier, name='SVR_atelier'),
    path('SVR_form/',views.SVR_form, name='SVR_form'),
    path('SVR_pred/',views.SVR_pred, name='SVR_pred'),

    #SVC
    path('SVC_atelier/',views.SVC_atelier, name='SVC_atelier'),
    path('SVC_form/',views.SVC_form, name='SVC_form'),
    path('SVC_pred/',views.SVC_pred, name='SVC_pred'),


    
        # Partie XG-Boost regression 
    path('DT_details/', views.DT_details, name='DT_details'),
    path('DT_reg_atelier/', views.DT_reg_atelier, name='DT_reg_atelier'),
    path('DT_reg_form/', views.DT_reg_form, name='DT_reg_form'),
    path('DT_reg_prediction/', views.DT_reg_prediction, name='DT_reg_prediction'),
    path('DT_cla_atelier/', views.DT_cla_atelier, name='DT_cla_atelier' ),
    path('DT_cla_form/', views.DT_cla_form, name='DT_cla_form'),
    path('DT_cla_prediction/', views.DT_cla_prediction, name='DT_cla_prediction'),
    
    



        path('informations/', views.informations, name='informations'),
        path('add_info_form/', views.add_info_form, name='add_info_form'),
        path('add_info/', views.add_info, name='add_info'),
        path('add_info_done/', views.add_info_done, name='add_info_done'),




        path('cleaning_form/', views.cleaning_form, name='cleaning_form'),
        path('cleaning_atelier/', views.cleaning_atelier, name='cleaning_atelier'),
        path('cleaning_proc/', views.cleaning_proc, name='cleaning_proc'),
        path('cleaning_done/', views.cleaning_done, name='cleaning_done'),




]



if settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)