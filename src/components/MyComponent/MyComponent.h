#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>
#include <sofa/type/Mat.h>
#include <sofa/type/vector.h>

using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::Data;

class MyComponent : public BaseObject {
   public:
    SOFA_CLASS(MyComponent, BaseObject);

    MyComponent();
    virtual ~MyComponent();

    Data<float> d_myparam;
};