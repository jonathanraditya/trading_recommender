def model_switcher(model_no, recurrent, train_df_input):
    '''Return compiled model based on model identifier'''
    if model_no == '132':
        return model_132(recurrent, train_df_input)
    elif model_no == '188':
        return model_188(recurrent, train_df_input)
    
def model_switcher_kt_build(model_no, hp, train_inputs, train_labels):
    if model_no == '263':
        return model_263_kt_build(hp, train_inputs, train_labels)
    elif model_no == '300':
        return model_300_kt_build(hp, train_inputs, train_labels)
    elif model_no == '301':
        return model_301_kt_build(hp, train_inputs, train_labels)
    elif model_no == '307':
        return model_307_kt_build(hp, train_inputs, train_labels)
    elif model_no == '308':
        return model_308_kt_build(hp, train_inputs, train_labels)
    elif model_no == '309':
        return model_309_kt_build(hp, train_inputs, train_labels)
    elif model_no == '310':
        return model_310_kt_build(hp, train_inputs, train_labels)
    elif model_no == '311':
        return model_311_kt_build(hp, train_inputs, train_labels)
    elif model_no == '312':
        return model_312_kt_build(hp, train_inputs, train_labels)
    elif model_no == '313':
        return model_313_kt_build(hp, train_inputs, train_labels)
    elif model_no == '314':
        return model_314_kt_build(hp, train_inputs, train_labels)
    elif model_no == '315':
        return model_315_kt_build(hp, train_inputs, train_labels)
    elif model_no == '316':
        return model_316_kt_build(hp, train_inputs, train_labels)
    elif model_no == '317':
        return model_317_kt_build(hp, train_inputs, train_labels)
    elif model_no == '318':
        return model_318_kt_build(hp, train_inputs, train_labels)
    elif model_no == '319':
        return model_319_kt_build(hp, train_inputs, train_labels)
    elif model_no == '320':
        return model_320_kt_build(hp, train_inputs, train_labels)
    else:
        print(f'Invalid model number. Check again. Passed value: {model_no}')
        pass
    
def model_263(train_inputs, train_labels):
    # MODEL 263: 0.6745-0.6149 #
    # tf.keras.layers.Lambda(lambda x: tf.math.multiply(x, np.array([0.3,0.7])))
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2])),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same'),
        tf.keras.layers.GRU(units=6, return_sequences=False),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005) # 
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # , run_eagerly=True
    ############
    return model

def model_263_kt(hp):
    '''model_263 kt version: 
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    first_conv_filters = hp.Int('first_conv_filters', min_value=8, max_value=64, step=8)
    second_conv_filters = hp.Int('second_conv_filters', min_value=16, max_value=128, step=16)
    second_conv_kernel_size = hp.Int('second_conv_kernel_size', min_value=1, max_value=15, step=2)
    second_conv_strides = hp.Int('second_conv_strides', min_value=1, max_value=5, step=1)
    third_conv_filters = hp.Int('third_conv_filters', min_value=16, max_value=128, step=16)
    third_conv_kernel_size = hp.Int('third_conv_kernel_size', min_value=1, max_value=20, step=3)
    third_conv_strides = hp.Int('third_conv_strides', min_value=1, max_value=5, step=1)
    recurrent_unit = hp.Int('recurrent_unit', min_value=6, max_value=60, step=6)
    dense_unit = hp.Int('dense_unit', min_value=12, max_value=60, step=12)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.00009, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=first_conv_filters, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=second_conv_filters, kernel_size=second_conv_kernel_size, strides=second_conv_strides, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=third_conv_filters, kernel_size=third_conv_kernel_size, strides=third_conv_strides, padding='causal')(X)
    X = tf.keras.layers.GRU(units=recurrent_unit, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=dense_unit, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_300_kt(hp):
    '''developed from model_263: 
    - additional C1D and recurrent layer
    - 2x more variable search range
    - widening lr range
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_filters_2 = hp.Int('c_filters_2', min_value=16, max_value=256, step=16)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=10, step=1)
    c_filters_3 = hp.Int('c_filters_3', min_value=16, max_value=256, step=16)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=10, step=1)
    c_filters_4 = hp.Int('c_filters_4', min_value=16, max_value=256, step=16)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=3)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=10, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_301_kt(hp):
    '''developed from model_300: 
    - add batch normalization and relu activation between convolutional layer
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_filters_2 = hp.Int('c_filters_2', min_value=16, max_value=256, step=16)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=10, step=1)
    c_filters_3 = hp.Int('c_filters_3', min_value=16, max_value=256, step=16)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=10, step=1)
    c_filters_4 = hp.Int('c_filters_4', min_value=16, max_value=256, step=16)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=3)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=10, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_302_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def bypass(X, X_bypass, operations):
        '''A conditional layer that let the optimizer
        choose to bypass, add, or concatenate'''
        # assert X.shape == X_bypass.shape
        if operations == 0:
            return X
        elif operations == 1:
            return tf.keras.layers.Add()([X, X_bypass])
        elif operations == 2:
            return tf.keras.layers.concatenate([X, X_bypass])
        
    bp_1 = hp.Choice('bp_1', [0,1])
    bp_2 = hp.Choice('bp_2', [0,1])
    bp_3 = hp.Choice('bp_3', [0,1])
    
    
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_kernel_1 = hp.Int('c_kernel_1', min_value=1, max_value=48, step=2)
    c_strides_1 = hp.Int('c_strides_1', min_value=1, max_value=4, step=1)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=4)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_1, strides=c_strides_1, padding='causal')(input_shape)
    X1 = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X1)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_2, strides=1, padding='causal')(X)
    X2 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X2, X1, operations=bp_1)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_3, strides=1, padding='causal')(X)
    X3 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X3, X2, operations=bp_2)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_4, strides=1, padding='causal')(X)
    X4 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X4, X3, operations=bp_3)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_303_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def bypass(X, X_bypass, operations):
        '''A conditional layer that let the optimizer
        choose to bypass, add, or concatenate'''
        # assert X.shape == X_bypass.shape
        if operations == 0:
            return X
        elif operations == 1:
            return tf.keras.layers.Add()([X, X_bypass])
        elif operations == 2:
            print(X.shape, X_bypass.shape)
            return tf.keras.layers.Concatenate(axis=1)([X, X_bypass])
        
    bp_1 = hp.Choice('bp_1', [0,2])
    bp_2 = hp.Choice('bp_2', [0,2])
    bp_3 = hp.Choice('bp_3', [0,2])
    
    
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=4, step=1)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=4, step=1)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=4)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=4, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X1 = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X1)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X2 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X2, X1, operations=bp_1)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X3 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X3, X2, operations=bp_2)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X4 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X4, X3, operations=bp_3)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_304_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    b1_filters = hp.Int('b1_filters', min_value=4, max_value=16, step=4)
    b2_filters = hp.Int('b2_filters', min_value=16, max_value=64, step=16)
    b3_filters = hp.Int('b3_filters', min_value=4, max_value=32, step=8)
    b1b2_recunits = hp.Int('b1b2_recunits', min_value=8, max_value=64, step=8)
    b3_recunits = hp.Int('b3_recunits', min_value=4, max_value=32, step=8)
    final_recunits = hp.Int('final_recunits', min_value=4, max_value=16, step=4)
    final_dunits = hp.Int('final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    # Block 1
    X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits)
    # Block 2
    X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
    # Addition block
    X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
    X = tf.keras.layers.Activation('relu')(X)
    # Block 3
    X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
    # Final layer
    X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
    X = tf.keras.layers.Dense(units=final_dunits, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_305_kt(hp):
    '''developed from model_304: 
    - split data sequence into sequence using prime number strides
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits):
        b1_filters = hp.Int(f'comp{component_num}_b1_filters', min_value=4, max_value=16, step=4)
        b2_filters = hp.Int(f'comp{component_num}_b2_filters', min_value=16, max_value=64, step=16)
        b3_filters = hp.Int(f'comp{component_num}_b3_filters', min_value=4, max_value=32, step=8)
        b1b2_recunits = hp.Int(f'comp{component_num}_1b2_recunits', min_value=8, max_value=64, step=8)
        b3_recunits = hp.Int(f'comp{component_num}_b3_recunits', min_value=4, max_value=32, step=8)
        final_recunits = hp.Int(f'comp{component_num}_final_recunits', min_value=4, max_value=16, step=4)
        # final_dunits = hp.Int(f'comp{component_num}_final_dunits', min_value=4, max_value=16, step=4)
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
        # Final layer
        X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits, activation='relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(recurrents)
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_306_kt(hp):
    '''developed from model_305: 
    - add component_strides to reduce number of components,
        thus reducing model complexity (the 305 ver take 5 minutes
        to compile)
    - add batch normalization in the end of superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits):
        b1_filters = hp.Int(f'comp{component_num}_b1_filters', min_value=4, max_value=16, step=4)
        b2_filters = hp.Int(f'comp{component_num}_b2_filters', min_value=16, max_value=64, step=16)
        b3_filters = hp.Int(f'comp{component_num}_b3_filters', min_value=4, max_value=32, step=8)
        b1b2_recunits = hp.Int(f'comp{component_num}_1b2_recunits', min_value=8, max_value=64, step=8)
        b3_recunits = hp.Int(f'comp{component_num}_b3_recunits', min_value=4, max_value=32, step=8)
        final_recunits = hp.Int(f'comp{component_num}_final_recunits', min_value=4, max_value=16, step=4)
        # final_dunits = hp.Int(f'comp{component_num}_final_dunits', min_value=4, max_value=16, step=4)
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
        # Final layer
        X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 3
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_307_kt(hp):
    '''developed from model_306: 
    - Keep component_strides=1, but maintaining complexity of
        each superblock by considering their total number of
        constituent.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_308_kt(hp):
    '''developed from model_307: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=0.75, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'{super_component_num}_comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hp.Int(f'{super_component_num}_superblock_final_dunits', min_value=2, max_value=6, step=2)
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=10):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
            
        
    recurrents = 120
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(recurrents,60))
    X = stack_superblock(input_shape, threshold=10)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_309_kt(hp):
    '''developed from model_307, optimized from model_308: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    - using threshold=2 and fraction=1. Compared to
        threshold=10 and fraction=0.75 in model_308.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'{super_component_num}_comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hp.Int(f'{super_component_num}_superblock_final_dunits', min_value=2, max_value=6, step=2)
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=2):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(recurrents,60))
    X = stack_superblock(input_shape, threshold=2)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_310_kt(hp):
    '''developed from model_307: 
    - 2-3x more layer units (with hope to increase performance)
        fraction 1 -> 3
        minimum_value 4 -> 8
        superblock_final_dunits 2/6/2 -> 8/24/4
        bdRNN 8 -> 32
        dense 8 -> 32
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=3, floor_fraction=0.25, num_steps=4, minimum_value=8):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=8, max_value=24, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32))(X)
    X = tf.keras.layers.Dense(units=32, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_311_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_rnnu = hp.Int('final_rnnu', min_value=8, max_value=48, step=8)
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=final_rnnu))(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model 

def model_312_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_313_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    Rev from model_312:
    - Early training show even worse loss: 0.9.
    - Try to stack conv1d 1 kernel
    - add model.summary()
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    final_c1d_1_filters = hp.Int('final_c1d_1_filters', min_value=4, max_value=8, step=2)
    final_c1d_2_filters = hp.Int('final_c1d_2_filters', min_value=4, max_value=16, step=4)
    final_c1d_2_kernels = hp.Int('final_c1d_2_kernels', min_value=1, max_value=7, step=1)
    final_c1d_2_strides = hp.Int('final_c1d_2_strides', min_value=1, max_value=3, step=1)
    final_c1d_3_filters = hp.Int('final_c1d_3_filters', min_value=4, max_value=16, step=4)
    final_c1d_3_kernels = hp.Int('final_c1d_3_kernels', min_value=1, max_value=7, step=1)
    final_c1d_3_strides = hp.Int('final_c1d_3_strides', min_value=1, max_value=3, step=1)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    # Stack multiple conv1d
    X = tf.keras.layers.Conv1D(filters=final_c1d_1_filters, kernel_size=1, strides=1)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_2_filters, kernel_size=final_c1d_2_kernels, strides=final_c1d_2_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_3_filters, kernel_size=final_c1d_3_kernels, strides=final_c1d_3_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_314_kt(hp):
    '''developed from model_307: 
    - Stacking superblock component together like block that stacked together.
    - Stacking difference with 308 & 309:
        - This 314 stack component before concatenation
            of superblock
        - The final concatenation is the same as 307, but
            with added component before that concatenation.
        - Desired effect:
            - Its possible that concatenation in 307 already
                abstract enough that additional abstraction
                just made the performance and learning curve
                going down.
            - Abstraction in prime component may give boost in
                performance by outputting more distinguishable
                value to later layer without shuffling the
                prime components even more.
            - This model also verify if stacking conv+rnn
                is possible. If not, try to stack more
                conv layer only to the component (next model).
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        X_bn_3, X_act_3 = superblock_component(X, superblock_hyp, component_num, first_node=False, final_node=True)
        
        X = tf.keras.layers.Dense(units=final_dunits)(X_act_3)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_315_kt(hp):
    '''developed from model_307: 
    - Difference with 307:
        - Add conv stack and additive layer in the superblock
        - Why?
            - Early kt search with 314 show very long model
                compilation although the network not complex enough
                in terms of parameters.
            - Need faster iteration between model
            - even if the model 314 is better compared to 307 & 310,
                comparable model with faster training time is needed.
            - It's still unknown wheter 314 complexity is enough to
                drive lower val_loss. Using this model variation as
                comparison can lower iteration time.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Block 1 + 2: Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Block 4
        X_bn_4, X_act_4 = block(X_act_3, superblock_hyp, superblock_hyp)
        # Block 3 + 4: Addition block
        X = tf.keras.layers.Add()([X_bn_3, X_bn_4])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 5
        X_bn_5, X_act_5 = block(X, superblock_hyp, superblock_hyp)
        
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_316_kt(hp):
    '''developed from model_314: 
    - Evaluation after ~2 epochs of training:
        - The gradient descent really slow, although
            the accuracy is superb with those high losses.
        - It's possible that the networks are to deep
            to significantly update earlier params.
        - Correction:
            - Remove third layer in superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_317_kt(hp):
    '''Benchmark model for multiple group size.
    Consist of:
    - recurrent + dense + concatenate + dense
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu)(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_317_kt_build(hp, a, b):
    '''Benchmark model for multiple group size.
    Consist of:
    - recurrent + dense + concatenate + dense
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu)(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_318_kt(hp):
    '''Deeper version of 317:
    - Add 2 conv layer before and after recurrent
    - add activation in final dense unit
    - Recurrent return_sequence=True
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_318_kt_build(hp, a, b):
    '''Deeper version of 317:
    - Add 2 conv layer before and after recurrent
    - add activation in final dense unit
    - Recurrent return_sequence=True
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_319_kt(hp):
    '''Modified from model 318
    - 3 recurrent component
    - addition block before final recurrent component
    - remove batch normalization after component dense unit
    - add activation after component dense unit
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        # 1st component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn1 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_319_kt_build(hp, a, b):
    '''Modified from model 318
    - 3 recurrent component
    - addition block before final recurrent component
    - remove batch normalization after component dense unit
    - add activation after component dense unit
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        # 1st component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn1 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_320_kt(hp):
    '''Modified from model 319
    - !! FIX 
        - flaw in input slice algorithm: remove `+1` in slicing formula.
        - Negligence that all Conv1D layer receive `input_slice` as
            previous layer. 
    - Add 3 recurrent preprocessor for input,
        and add them up as second layer component input.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    input_preprocessors = 3
    processor_features = base_features / input_preprocessors
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        input_preprocessor_results = []
        for j in range(input_preprocessors):
            input_preprocessor = input_shape[:,:,int(base_features*i + j*processor_features):int(base_features*i + (j+1)*processor_features)]
            # 1st component layer
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_preprocessor)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)

            X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            input_preprocessor_results.append(X)
        
        X_bn1 = tf.keras.layers.Add()(input_preprocessor_results)        
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_320_kt_build(hp, a, b):
    '''Modified from model 319
    - !! FIX 
        - flaw in input slice algorithm: remove `+1` in slicing formula.
        - Negligence that all Conv1D layer receive `input_slice` as
            previous layer. 
    - Add 3 recurrent preprocessor for input,
        and add them up as second layer component input.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    input_preprocessors = 3
    processor_features = base_features / input_preprocessors
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        input_preprocessor_results = []
        for j in range(input_preprocessors):
            input_preprocessor = input_shape[:,:,int(base_features*i + j*processor_features):int(base_features*i + (j+1)*processor_features)]
            # 1st component layer
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_preprocessor)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)

            X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            input_preprocessor_results.append(X)
        
        X_bn1 = tf.keras.layers.Add()(input_preprocessor_results)        
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_321_kt(hp):
    '''Benchmark model for world data model.
    
    conv + lstm
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    c_filters = hp.Int('c_filters', min_value=8, max_value=64, step=8)
    r_units = hp.Int('r_units', min_value=32, max_value=128, step=16)
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=c_filters, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.LSTM(units=r_units, return_sequences=False)(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_300_kt_build(hp, train_inputs, train_labels):
    '''developed from model_263: 
    - additional C1D and recurrent layer
    - 2x more variable search range
    - widening lr range
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1, c_filters_2, c_kernel_2, c_strides_2, c_filters_3, c_kernel_3, c_strides_3, c_filters_4, c_kernel_4, c_strides_4, r_unit_1, r_unit_2, d_unit_1, lr = hp
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_301_kt_build(hp, train_inputs, train_labels):
    '''developed from model_300: 
    - add batch normalization and relu activation between convolutional layer
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1, c_filters_2, c_kernel_2, c_strides_2, c_filters_3, c_kernel_3, c_strides_3, c_filters_4, c_kernel_4, c_strides_4, r_unit_1, r_unit_2, d_unit_1, lr = hp
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_307_kt_build(hp, train_inputs, train_labels):
    '''developed from model_306: 
    - Keep component_strides=1, but maintaining complexity of
        each superblock by considering their total number of
        constituent.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_308_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', '0_superblock_final_dunits', '0_comp2', '0_comp3', '0_comp5', '0_comp7', '0_comp11', '0_comp13', '0_comp17', '0_comp19', '0_comp23', '0_comp29', '0_comp31', '0_comp37', '0_comp41', '0_comp43', '0_comp47', '0_comp53', '0_comp59', '0_comp61', '0_comp67', '0_comp71', '0_comp73', '0_comp79', '0_comp83', '0_comp89', '0_comp97', '0_comp101', '0_comp103', '0_comp107', '0_comp109', '0_comp113', '1_superblock_final_dunits', '1_comp2', '1_comp3', '1_comp5', '1_comp7', '1_comp11', '1_comp13', '1_comp17', '1_comp19', '1_comp23', '1_comp29']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=0.75, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'{super_component_num}_comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hyperparameters[f'{super_component_num}_superblock_final_dunits']
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=10):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hyperparameters['lr']
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = stack_superblock(input_shape, threshold=10)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_309_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307, optimized from model_308: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    - using threshold=2 and fraction=1. Compared to
        threshold=10 and fraction=0.75 in model_308.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)''' 
    
    keys = ['lr', '0_superblock_final_dunits', '0_comp2', '0_comp3', '0_comp5', '0_comp7', '0_comp11', '0_comp13', '0_comp17', '0_comp19', '0_comp23', '0_comp29', '0_comp31', '0_comp37', '0_comp41', '0_comp43', '0_comp47', '0_comp53', '0_comp59', '0_comp61', '0_comp67', '0_comp71', '0_comp73', '0_comp79', '0_comp83', '0_comp89', '0_comp97', '0_comp101', '0_comp103', '0_comp107', '0_comp109', '0_comp113', '1_superblock_final_dunits', '1_comp2', '1_comp3', '1_comp5', '1_comp7', '1_comp11', '1_comp13', '1_comp17', '1_comp19', '1_comp23', '1_comp29', '2_superblock_final_dunits', '2_comp2', '2_comp3', '2_comp5', '2_comp7', '3_superblock_final_dunits', '3_comp2', '3_comp3']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'{super_component_num}_comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hyperparameters[f'{super_component_num}_superblock_final_dunits']
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=2):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hyperparameters['lr']
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = stack_superblock(input_shape, threshold=2)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_310_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - 2-3x more layer units (with hope to increase performance)
        fraction 1 -> 3
        minimum_value 4 -> 8
        superblock_final_dunits 2/6/2 -> 8/24/4
        bdRNN 8 -> 32
        dense 8 -> 32
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32))(X)
    X = tf.keras.layers.Dense(units=32, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_311_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_rnnu', 'final_denseu', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_rnnu = hyperparameters['final_rnnu']
    final_denseu = hyperparameters['final_denseu']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=final_rnnu))(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_312_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_denseu', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_313_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    Rev from model_312:
    - Early training show even worse loss: 0.9.
    - Try to stack conv1d 1 kernel
    - add model.summary()
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_denseu', 'final_c1d_1_filters', 'final_c1d_2_filters', 'final_c1d_2_kernels', 'final_c1d_2_strides', 'final_c1d_3_filters', 'final_c1d_3_kernels', 'final_c1d_3_strides', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    final_c1d_1_filters = hyperparameters['final_c1d_1_filters']
    final_c1d_2_filters = hyperparameters['final_c1d_2_filters']
    final_c1d_2_kernels = hyperparameters['final_c1d_2_kernels']
    final_c1d_2_strides = hyperparameters['final_c1d_2_strides']
    final_c1d_3_filters = hyperparameters['final_c1d_3_filters']
    final_c1d_3_kernels = hyperparameters['final_c1d_3_kernels']
    final_c1d_3_strides = hyperparameters['final_c1d_3_strides']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    # Stack multiple conv1d
    X = tf.keras.layers.Conv1D(filters=final_c1d_1_filters, kernel_size=1, strides=1)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_2_filters, kernel_size=final_c1d_2_kernels, strides=final_c1d_2_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_3_filters, kernel_size=final_c1d_3_kernels, strides=final_c1d_3_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_314_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Stacking superblock component together like block that stacked together.
    - Stacking difference with 308 & 309:
        - This 314 stack component before concatenation
            of superblock
        - The final concatenation is the same as 307, but
            with added component before that concatenation.
        - Desired effect:
            - Its possible that concatenation in 307 already
                abstract enough that additional abstraction
                just made the performance and learning curve
                going down.
            - Abstraction in prime component may give boost in
                performance by outputting more distinguishable
                value to later layer without shuffling the
                prime components even more.
            - This model also verify if stacking conv+rnn
                is possible. If not, try to stack more
                conv layer only to the component (next model).
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        X_bn_3, X_act_3 = superblock_component(X, superblock_hyp, component_num, first_node=False, final_node=True)
        
        X = tf.keras.layers.Dense(units=final_dunits)(X_act_3)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_315_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Difference with 307:
        - Add conv stack and additive layer in the superblock
        - Why?
            - Early kt search with 314 show very long model
                compilation although the network not complex enough
                in terms of parameters.
            - Need faster iteration between model
            - even if the model 314 is better compared to 307 & 310,
                comparable model with faster training time is needed.
            - It's still unknown wheter 314 complexity is enough to
                drive lower val_loss. Using this model variation as
                comparison can lower iteration time.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Block 1 + 2: Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Block 4
        X_bn_4, X_act_4 = block(X_act_3, superblock_hyp, superblock_hyp)
        # Block 3 + 4: Addition block
        X = tf.keras.layers.Add()([X_bn_3, X_bn_4])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 5
        X_bn_5, X_act_5 = block(X, superblock_hyp, superblock_hyp)
        
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_316_kt_build(hp, train_inputs, train_labels):
    '''developed from model_314: 
    - Evaluation after ~2 epochs of training:
        - The gradient descent really slow, although
            the accuracy is superb with those high losses.
        - It's possible that the networks are to deep
            to significantly update earlier params.
        - Correction:
            - Remove third layer in superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model