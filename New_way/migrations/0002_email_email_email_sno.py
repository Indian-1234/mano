# Generated by Django 4.1.6 on 2023-04-18 05:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('New_way', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='email',
            name='EMAIL',
            field=models.EmailField(default=9, max_length=254),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='email',
            name='SNO',
            field=models.TextField(default='yuhyu'),
            preserve_default=False,
        ),
    ]