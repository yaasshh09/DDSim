import numpy as np, pytest
from  ddsim.core.field import Field,  Location,   ScalingState
from ddsim.core.scaling import ScaleFactors



@pytest.fixture


def scale( ) ->   ScaleFactors  :

    return ScaleFactors.for_silicon( )




def psi_physical(values :  list[float] |None  =None) -> Field :

    return Field(values or[0.0,0.5,1.0],'V',ScalingState.PHYSICAL,Location.NODE)


def psi_scaled(values : list[float]|  None =  None)-> Field:
    return Field(values or[0.0, 19.3, 38.7], "V", ScalingState.SCALED, Location.NODE)

def test_field_carries_data_unit_scaling_and_location()-> None:
    kk =Field([1.0,2.0],'cm^-3',ScalingState.PHYSICAL,Location.NODE);assert kk.unit == "cm^-3"
    assert kk.scaling is ScalingState.PHYSICAL
    assert kk.location  is  Location.NODE
def test_data_is_a_plain_numpy_array_for_hot_loops() -> None:
    m2=psi_physical();assert isinstance(m2.data,np.ndarray)
    assert m2.data.dtype  ==np.float64
    np.testing.assert_array_equal (m2.data,  [  0.0,  0.5,  1.0] )
def  test_shape_and_len_delegate_to_the_array( ) ->  None  :
    m2  = psi_physical(  ); assert  m2.shape ==  (3 ,  );  assert m2.size== 3
    assert  len(  m2  )  ==   3



def test_field_accepts_an_optional_name()   ->   None  :
    res = Field ( [1.0 ],   'V' , ScalingState.PHYSICAL,   Location.NODE , name =  "psi" )
    assert res.name==  'psi'



def test_repr_shows_unit_scaling_and_location(  )  ->  None :
    i  =repr(psi_scaled())
    assert "V" in i
    assert "SCALED" in i
    assert 'NODE' in i

def  test_add_with_matching_metadata_succeeds()   -> None  :
    aa  =   psi_physical (  )  +   psi_physical ( ) ; np.testing.assert_allclose(aa.data, [0.0, 1.0, 2.0])
    assert aa.unit =="V"
    assert  aa.scaling is  ScalingState.PHYSICAL
    assert aa.location is  Location.NODE



def test_subtract_with_matching_metadata_succeeds() ->None :
    f = psi_physical()-psi_physical()


    np.testing.assert_allclose(f.data,[0.0,0.0,0.0])

def test_add_different_scaling_state_raises ( )  ->  None  :

    with  pytest.raises(  ValueError ,   match  =   'scaling')   :
        psi_scaled() +psi_physical()
def test_add_different_location_raises()  -> None :
    dat =psi_physical(); t= Field([1.0, 2.0], 'V', ScalingState.PHYSICAL, Location.EDGE)
    with  pytest.raises ( ValueError ,  match  =   "location" )   :
        dat+ t

def test_add_different_unit_raises() -> None :
    s =psi_physical()
    v = Field([1.0, 2.0, 3.0], 'cm^-3', ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError, match = "unit")  :
        s + v



def test_subtract_different_scaling_state_raises()  -> None  :
    with pytest.raises(ValueError, match ='scaling'):
        psi_scaled( )   -  psi_physical (  )

def test_subtract_different_location_raises()-> None :
    kk =  Field([1.0, 2.0], 'V', ScalingState.PHYSICAL, Location.EDGE)
    with pytest.raises(ValueError, match ="location") :
        psi_physical()-  kk


def test_subtract_different_unit_raises() -> None  :
    h  =  Field([  1.0,   2.0, 3.0],  'cm^-3',  ScalingState.PHYSICAL,  Location.NODE)
    with pytest.raises(ValueError, match=  'unit') :
        psi_physical()- h


def test_add_a_bare_scalar_raises()->None:

    with pytest.raises ( TypeError )  :
        psi_physical()  + 1.0


def test_add_a_bare_array_raises ( )  ->  None :
    with pytest.raises (TypeError )  :
        psi_physical() + np.array([1.0, 2.0, 3.0])


def test_add_mismatched_length_raises() ->  None :
    u   =  Field ( [1.0 ] , 'V',   ScalingState.PHYSICAL ,  Location.NODE  )
    with  pytest.raises( ValueError,  match  =   'length' )  :
        psi_physical() + u


def test_negation_preserves_metadata()-> None :
    f   = -   psi_physical( )
    np.testing.assert_allclose( f.data ,   [  0.0,   -   0.5,  -   1.0 ]  )
    assert f.unit  == 'V'
    assert f.scaling is ScalingState.PHYSICAL


def test_multiply_combines_unit_strings(  )  ->  None :
    ok= psi_physical()
    m= Field([2.0,2.0,2.0],'cm^-3',ScalingState.PHYSICAL,Location.NODE);  assert(ok *m).unit=='V*cm^-3'


def test_divide_combines_unit_strings()->None:
    ys  = psi_physical()
    e  = Field([2.0, 2.0, 2.0], "cm", ScalingState.PHYSICAL, Location.NODE)
    assert(ys  / e).unit== "V/cm"


def test_divide_by_a_compound_unit_parenthesises_it ()  ->  None   :
    v=psi_physical()

    i = Field([2.0, 2.0, 2.0], 'cm^2/s', ScalingState.PHYSICAL, Location.NODE)
    assert (  v  /  i  ).unit ==  "V/(cm^2/s)"

def test_multiply_by_dimensionless_preserves_the_other_unit()->None :
    a2=psi_physical()


    t =Field([2.0, 2.0, 2.0], '1', ScalingState.PHYSICAL, Location.NODE)
    assert(a2 * t).unit=="V"
    assert ( t *  a2  ).unit  ==   "V"

def test_divide_identical_units_gives_dimensionless()-> None:
    res=psi_physical([1.0, 2.0, 4.0])
    assert(res/ res).unit ==  "1"



def test_divide_by_dimensionless_preserves_the_unit()  ->None:
    t2 = Field([2.0,2.0,2.0],'1',ScalingState.PHYSICAL,Location.NODE)
    assert ( psi_physical(  )  /  t2  ).unit  ==   'V'

def test_multiply_different_scaling_state_raises()-> None :
    d=Field([1.0,1.0,1.0],"cm^-3",ScalingState.SCALED,Location.NODE)
    with pytest.raises(ValueError, match  = "scaling"):
        psi_physical()*d




def test_multiply_different_location_raises()->None :
    b =Field([1.0,2.0],"cm^-3",ScalingState.PHYSICAL,Location.EDGE)
    with pytest.raises(ValueError, match   = 'location')  :

        psi_physical()* b


def test_divide_different_scaling_state_raises() -> None :
    m   = Field( [  1.0,   1.0, 1.0  ], 'cm',  ScalingState.SCALED, Location.NODE)
    with pytest.raises(ValueError,match= 'scaling'):
        psi_physical() /  m


def test_multiply_by_a_python_scalar_preserves_the_unit() ->None:
    res=psi_physical()* 2.0
    np.testing.assert_allclose(res.data,[0.0,1.0,2.0])
    assert res.unit== "V"

def test_right_multiply_by_a_python_scalar_works() -> None :
    s2 =   2.0 *  psi_physical()


    np.testing.assert_allclose(s2.data, [0.0, 1.0, 2.0])



def test_divide_by_a_python_scalar_preserves_the_unit() ->  None:
    e = psi_physical() /2.0
    np.testing.assert_allclose(e.data, [0.0, 0.25, 0.5])
    assert e.unit== 'V'

def test_to_scaled_uses_the_scale_factors(scale: ScaleFactors)->None:
    w2   =   psi_physical().to_scaled(scale  )
    assert w2.scaling is ScalingState.SCALED


    np.testing.assert_allclose(w2.data,np.array([0.0,0.5,1.0]) / scale.psi_0)

def  test_to_physical_uses_the_scale_factors(scale  :  ScaleFactors )  -> None  :
    b   = Field( [1.0  ] ,   'V', ScalingState.SCALED,   Location.NODE  )
    xs  = b.to_physical(scale)
    assert xs.scaling is ScalingState.PHYSICAL
    np.testing.assert_allclose( xs.data, [ scale.psi_0 ] )
def  test_to_scaled_is_not_hardcoded_to_a_single_factor (scale  : ScaleFactors,)   -> None  :
    s = Field([1e16], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    y  =  ScaleFactors.for_silicon(  C_0   = 1e18 )
    assert s.to_scaled(scale).data[0] != s.to_scaled(y).data[0]
    np.testing.assert_allclose(s.to_scaled(y).data,[1e16 /1e18])

def test_round_trip_to_scaled_and_back_preserves_data(scale: ScaleFactors)->None:
    h=  psi_physical([-  1.5, 0.0, 3.25])
    cnt  = h.to_scaled(scale).to_physical(scale)
    np.testing.assert_allclose(cnt.data,h.data,rtol=1e-14,atol=0.0)



def test_to_scaled_on_an_already_scaled_field_raises()-> None :
    with pytest.raises(ValueError,match ='already') :
        psi_scaled( ).to_scaled(  ScaleFactors.for_silicon (  )  )


def test_to_physical_on_an_already_physical_field_raises()-> None  :
    with  pytest.raises(ValueError,   match  =  'already'  ) :
        psi_physical( ).to_physical(ScaleFactors.for_silicon ( )  )


def test_conversion_preserves_unit_and_location(scale: ScaleFactors) ->None :
    z=  Field([1.0, 2.0], 'A/cm^2', ScalingState.PHYSICAL, Location.EDGE)
    vv  =  z.to_scaled(  scale  )
    assert vv.unit   ==  'A/cm^2'
    assert vv.location is Location.EDGE




def test_scaling_state_cannot_be_reassigned()->None:

    c2  = psi_physical ( )
    with pytest.raises(AttributeError):
        c2.scaling=ScalingState.SCALED




def test_unit_cannot_be_reassigned()->None :
    arr  = psi_physical()
    with pytest.raises(AttributeError):
        arr.unit=  "cm^-3"
def test_field_does_not_expose_array_protocol() ->  None :


    assert not hasattr(Field, '__array__')

def test_refuses_to_add_a_scaled_potential_to_a_physical_one( )  -> None :
    w  = Field(  [ 38.7 ] ,   'V',   ScalingState.SCALED , Location.NODE)
    k  =  Field([  1.0 ], 'V',   ScalingState.PHYSICAL ,  Location.NODE )
    with pytest.raises(ValueError) :
        w +k
